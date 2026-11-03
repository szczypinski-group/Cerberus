"""Step 2: generate condensation products with reaction SMARTS.

Every product carries a site tag (which aldehyde carbon met which NH2), so
molecules with several reactive groups give several, distinguishable products.
New C=N bonds are enumerated as E and Z. All stereoisomers of one molecule
share one atom order, so their conformer ensembles can be pooled (CENSO).
"""

from __future__ import annotations

from collections.abc import Iterator

from rdkit import Chem
from rdkit.Chem import AllChem, rdCIPLabeler

from predcc_dft import chem
from predcc_dft.io_utils import Config

CN_C, CN_N = (
    1,
    3,
)  # map numbers of the carbonyl C and reacting N in the templates


def _site_tag(ald_site: int, nuc_site: int) -> str:
    return f"ald:C{ald_site}|nuc:N{nuc_site}"


def _canonical_order(mol: Chem.Mol) -> Chem.Mol:
    """Renumber heavy atoms by a stereo-blind canonical ranking.

    E and Z isomers of the same graph receive the same atom order, so their
    geometries can be concatenated into one ensemble.
    """
    ranks = list(
        Chem.CanonicalRankAtoms(mol, breakTies=True, includeChirality=False)
    )
    order = sorted(range(mol.GetNumAtoms()), key=lambda i: ranks[i])
    out = Chem.RenumberAtoms(mol, order)
    # Refresh stereo perception so explicitly set E/Z survives the renumbering.
    Chem.AssignStereochemistry(out, force=True, cleanIt=False)
    return out


def _stereo_label(mol: Chem.Mol) -> str:
    """CIP labels of all stereo double bonds, e.g. 'E', 'EZ'; or 'none'."""
    m = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(m)
    codes = [
        b.GetProp("_CIPCode")
        for b in m.GetBonds()
        if b.GetBondType() == Chem.BondType.DOUBLE and b.HasProp("_CIPCode")
    ]
    return "".join(codes) if codes else "none"


def _set_cn_stereo(
    mol: Chem.Mol, c_idx: int, n_idx: int, stereo: Chem.BondStereo
) -> Chem.Mol:
    """Copy of mol with the C=N bond set to the given stereo."""
    m = Chem.RWMol(mol)
    bond = m.GetBondBetweenAtoms(c_idx, n_idx)
    begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    b_nbr = [
        a.GetIdx()
        for a in m.GetAtomWithIdx(begin).GetNeighbors()
        if a.GetIdx() != end
    ]
    e_nbr = [
        a.GetIdx()
        for a in m.GetAtomWithIdx(end).GetNeighbors()
        if a.GetIdx() != begin
    ]
    bond.SetStereoAtoms(b_nbr[0], e_nbr[0])
    bond.SetStereo(stereo)
    return m.GetMol()


def enumerate_ez(mol: Chem.Mol, enumerate_new: bool = True) -> list[Chem.Mol]:
    """E and Z isomers of the newly formed C=N (atoms tagged _cn = 1/3)."""
    c_idx = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.HasProp("_cn") and a.GetIntProp("_cn") == CN_C
    )
    n_idx = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.HasProp("_cn") and a.GetIntProp("_cn") == CN_N
    )
    if not enumerate_new:
        return [mol]
    isomers = []
    for stereo in (Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOCIS):
        isomers.append(_fresh(_set_cn_stereo(mol, c_idx, n_idx, stereo)))
    return isomers


def _run_template(
    linkage: str, ald: Chem.Mol, nuc: Chem.Mol
) -> Iterator[tuple[Chem.Mol, int, int]]:
    """Yield (product, aldehyde site, nucleophile site) for one template."""
    rxn = AllChem.ReactionFromSmarts(chem.REACTION_SMARTS[linkage])
    for prods in rxn.RunReactants((ald, nuc)):
        prod = Chem.RWMol(prods[0])
        ald_site = nuc_site = None
        for atom in prod.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.HasProp(
                "_cn"
            ):  # inherited from a hydrazone reactant (azine route)
                atom.ClearProp("_cn")
            if atom.HasProp("old_mapno"):
                mapno = atom.GetIntProp("old_mapno")
                if mapno == CN_C:
                    atom.SetIntProp("_cn", CN_C)
                    ald_site = atom.GetIntProp("react_atom_idx")
                elif mapno == CN_N:
                    atom.SetIntProp("_cn", CN_N)
                    nuc_site = atom.GetIntProp("react_atom_idx")
        try:
            prod = prod.GetMol()
            Chem.SanitizeMol(prod)
        except Exception:  # noqa: BLE001 - invalid product from an odd match
            continue
        if ald_site is None or nuc_site is None:
            continue
        yield _fresh(prod), ald_site, nuc_site


def _fresh(prod: Chem.Mol) -> Chem.Mol:
    """Round-trip through SMILES so stereo can be set; keeps the C=N tags."""
    tmp = Chem.Mol(prod)
    for atom in tmp.GetAtoms():
        atom.SetAtomMapNum(
            atom.GetIntProp("_cn") if atom.HasProp("_cn") else 0
        )
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(tmp))
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            atom.SetIntProp("_cn", atom.GetAtomMapNum())
            atom.SetAtomMapNum(0)
    return mol


def _resolve_key_collisions(isomers: dict[str, dict]) -> None:
    """E/Z (or tautomers) sharing a standard InChIKey get -E/-Z (SOP 2.2)."""
    by_key: dict[str, list[dict]] = {}
    for iso in isomers.values():
        base = iso.setdefault("inchikey_standard", iso["inchikey"])
        by_key.setdefault(base, []).append(iso)
    for base, group in by_key.items():
        if len(group) > 1:
            for iso in group:
                iso["inchikey"] = f"{base}-{iso['stereo']}"
                iso["key_collision"] = True


class ProductSet:
    """Condensation products by compound key, with origins and isomers."""

    def __init__(self, enumerate_ez_flag: bool = True) -> None:
        """Start an empty set; enumerate E and Z unless told otherwise."""
        self.enumerate = enumerate_ez_flag
        self.compounds: dict[str, dict] = {}

    def add(
        self,
        prod: Chem.Mol,
        linkage: str,
        ald: dict,
        nuc: dict,
        ald_site: int,
        nuc_site: int,
    ) -> str:
        """Add one template product (E and Z) and return its compound key."""
        origin = {
            "aldehyde": ald["compound"],
            "amine": nuc["compound"],
            "aldehyde_site": ald_site,
            "amine_site": nuc_site,
            "site_tag": _site_tag(ald_site, nuc_site),
            "multi_site": len(ald["sites"]) > 1 or len(nuc["sites"]) > 1,
        }
        isomers = enumerate_ez(prod, self.enumerate)
        ckey = chem.compound_key(chem.inchikey(isomers[0]))
        comp = self.compounds.get(ckey)
        if comp is None:
            comp = self.compounds[ckey] = {
                "compound": ckey,
                "role": "imine",
                "linkage": linkage,
                "name": f"{ald['name']} + {nuc['name']} ({linkage})",
                "smiles": chem.canonical(prod, isomeric=False),
                "origins": [],
                "isomers": {},
            }
        if origin not in comp["origins"]:
            comp["origins"].append(origin)
        for iso in isomers:
            iso = _canonical_order(iso)
            smi = chem.canonical(iso)
            if smi in comp["isomers"]:
                continue
            cn_c = next(
                a.GetIdx()
                for a in iso.GetAtoms()
                if a.HasProp("_cn") and a.GetIntProp("_cn") == CN_C
            )
            cn_n = next(
                a.GetIdx()
                for a in iso.GetAtoms()
                if a.HasProp("_cn") and a.GetIntProp("_cn") == CN_N
            )
            comp["isomers"][smi] = {
                "mol": iso,
                "smiles": smi,
                "inchikey": chem.inchikey(iso),
                "inchi": chem.inchi(iso),
                "stereo": _stereo_label(iso),
                "cn_bond": [cn_c, cn_n],
            }
        _resolve_key_collisions(comp["isomers"])
        return ckey


def generate(
    aldehydes: list[dict], nucleophiles: list[dict], cfg: Config
) -> ProductSet:
    """Run every allowed template on every aldehyde x nucleophile pair.

    Hydrazones with a free NH2 then condense with a second aldehyde (azines).
    """
    allowed = set(cfg["products"]["linkages"])
    products = ProductSet(cfg["products"].get("enumerate_ez", True))
    for ald in aldehydes:
        for nuc in nucleophiles:
            for cls in nuc["classes"]:
                linkage = chem.CLASS_TO_LINKAGE[cls]
                if linkage not in allowed:
                    continue
                for prod, a_site, n_site in _run_template(
                    linkage, ald["mol"], nuc["mol"]
                ):
                    products.add(prod, linkage, ald, nuc, a_site, n_site)

    # Azines: hydrazones with a free NH2 condense with a second aldehyde.
    if "azine" in allowed:
        hydrazone_patt = Chem.MolFromSmarts(
            chem.NUCLEOPHILE_CLASSES["hydrazone"]
        )
        hydrazones = []
        for comp in list(products.compounds.values()):
            if comp["linkage"] != "hydrazone":
                continue
            for iso in comp["isomers"].values():
                if iso["mol"].HasSubstructMatch(hydrazone_patt):
                    hydrazones.append(
                        {
                            "compound": comp["compound"],
                            "name": comp["name"],
                            "mol": iso["mol"],
                            "sites": chem.match_sites(
                                iso["mol"],
                                chem.NUCLEOPHILE_CLASSES["hydrazone"],
                            ),
                        }
                    )
        for ald in aldehydes:
            for hyd in hydrazones:
                for prod, a_site, n_site in _run_template(
                    "azine", ald["mol"], hyd["mol"]
                ):
                    products.add(prod, "azine", ald, hyd, a_site, n_site)
    return products
