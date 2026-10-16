"""Species boundaries and exact mouse reference tests without model weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest

from tcr_workbench.backends import pmhc_decoder
from tcr_workbench.backends.mhc_context import load_reference
from tcr_workbench.hla import compare_hla, normalize_hla
from tcr_workbench.ingest import read_receptors
from tcr_workbench.prediction import _Pair, _input_frame, export_decoder_input
from tcr_workbench.species import biological_fingerprint, model_mhc, normalize_mhc, verify_biological_fingerprint

MOUSE_PAIR = dict(trav="TRAV13-1*01", traj="TRAJ48*01", cdr3a="CAMNYGNEKITF",
                  trbv="TRBV20*01", trbj="TRBJ2-7*01", cdr3b="CGARRDWGSSYEQYF")


@pytest.mark.parametrize("raw,canonical", [("H2-Kb", "H-2-Kb"), ("H-2Db", "H-2-Db"),
                                             ("H-2-IAb", "H-2-IAb"), ("h-2-iak", "H-2-IAk")])
def test_mouse_identifiers_are_literal_contexts(raw, canonical):
    assert normalize_hla(raw) == canonical
    assert normalize_mhc(raw, "mouse") == canonical
    assert compare_hla(raw, canonical).status == "compatible"


@pytest.mark.parametrize("hla,species", [("H-2-Kb", "human"), ("HLA-A*02:01", "mouse")])
def test_species_never_silently_substitutes_mhc(hla, species):
    with pytest.raises(ValueError, match="does not match"):
        model_mhc(hla, species)
    with pytest.raises(ValueError, match="does not match"):
        _Pair.model_validate(dict(**MOUSE_PAIR, species=species, hla=hla, name="one", peptide="SIINFEKL"))


def test_mouse_pair_validation_and_human_default():
    pair = _Pair.model_validate(dict(**MOUSE_PAIR, species="mouse", hla="H2-Kb", name="one", peptide="SIINFEKL"))
    assert pair.hla == "H-2-Kb" and pair.species == "mouse"
    with pytest.raises(ValueError, match="does not match"):
        _Pair.model_validate(dict(**MOUSE_PAIR, hla="H-2-Kb", name="one", peptide="SIINFEKL"))
    assert compare_hla("H-2-Kb", "HLA-A*02:01").status == "incompatible"
    assert compare_hla("H-2-Kb", "H-2-Kd").status == "incompatible"


@pytest.mark.parametrize("gene", [
    "TRAV13-4/DV7", "TRAV14D-3/DV8", "TRAV15-1/DV6-1", "TRAV15-2/DV6-2",
    "TRAV15D-1/DV6D-1", "TRAV15D-2/DV6D-2", "TRAV16D/DV11", "TRAV21/DV12",
    "TRAV4-4/DV10", "TRAV6-7/DV9",
])
def test_mouse_dual_use_gene_names_remain_single_genes(tmp_path, gene):
    # All ten base names occur in the downloaded IMGT MOUSE/TRA.fasta.
    # Allele membership remains the worker's exact-reference responsibility.
    for value in (gene, gene + "*01"):
        components = {**MOUSE_PAIR, "trav": value}
        pair = _Pair.model_validate(dict(**components, species="mouse", hla="H-2-Kb", name="one", peptide="SIINFEKL"))
        assert pair.trav == value
        source = tmp_path / "input.csv"
        pl.DataFrame([dict(**components, cell_id="cell1", donor_id="mouse1")]).write_csv(source)
        receptors = read_receptors(source, species="mouse").receptors
        assert receptors["trav"][0] == value
        output = tmp_path / "pairs.csv"
        result = export_decoder_input(receptors, pl.DataFrame({"peptide": ["SIINFEKL"], "hla": ["H-2-Kb"]}), output, species="mouse")
        assert result["exported_pairs"] == 1
        assert _input_frame(output, species="mouse")["trav"][0] == value


@pytest.mark.parametrize("gene", ["TRAV14D-3/TRAV15-1", "TRAV14D-3/DV8/TRAV15-1",
                                 "TRAV15D-1/DV6D-1*01/*02", "TRAV15D-1/DV6D-1,TRAV21",
                                 "TRAV15D-1/DV6D-1|TRAV21", "TRAV15D-1/DV6D-1;TRAV21"])
def test_mouse_slash_gene_alternatives_are_not_dual_use(gene):
    with pytest.raises(ValueError, match="one unambiguous gene"):
        _Pair.model_validate(dict(**{**MOUSE_PAIR, "trav": gene}, species="mouse", hla="H-2-Kb", name="one", peptide="SIINFEKL"))


def test_bundled_mouse_reference_has_exact_classI_and_classII_chains():
    fingerprint = biological_fingerprint("mouse")
    reference = load_reference(fingerprint["mhc_reference_path"])
    molecules = reference["molecules"]
    assert set(molecules) == {"H-2-Kb", "H-2-Db", "H-2-IAb", "H-2-IAk"}
    assert len(molecules["H-2-Kb"]["HLA_a"]) == 348
    assert len(molecules["H-2-Db"]["HLA_a"]) == 338
    assert len(molecules["H-2-Kb"]["HLA_b"]) == 99
    assert molecules["H-2-Kb"]["HLA_b"] == molecules["H-2-Db"]["HLA_b"]
    assert len(molecules["H-2-IAb"]["HLA_a"]) == 233
    assert len(molecules["H-2-IAb"]["HLA_b"]) == 238
    assert "P01887" in molecules["H-2-Kb"]["source"]
    assert "P14483" in molecules["H-2-IAb"]["source"]
    assert "CC BY 4.0" in reference["source"]
    verify_biological_fingerprint(fingerprint)


def test_mouse_v2_scoring_adapter_uses_reported_leader_alleles():
    pytest.importorskip("DecoderTCR")
    from DecoderTCR.reconstruct.components import _mouse_leaders
    from tcr_workbench.backends.reconstruct_worker import scoring_chain_adapter

    leaders = _mouse_leaders()
    gene, leader = next(iter(leaders.items()))
    prepared = scoring_chain_adapter("v2", "mouse")({
        "TCR_a": leader + "AC", "TCR_b": leader + "DE", "ok": True, "reason": "",
        "TRA_leader": gene + "(L)", "TRB_leader": gene,
    })
    assert prepared["ok"] and prepared["TCR_a"] == "AC" and prepared["TCR_b"] == "DE"
    rejected = scoring_chain_adapter("v2", "mouse")({
        "TCR_a": leader + "AC", "TCR_b": leader + "DE", "ok": True, "reason": "",
        "TRA_leader": "missing*01(L)", "TRB_leader": gene,
    })
    assert not rejected["ok"] and "No verified mouse leader" in rejected["reason"]


def test_real_mouse_v2_reconstruction_matches_public_tcr_and_custom_mhc(tmp_path):
    pytest.importorskip("DecoderTCR")
    from Stitchr import stitchrfunctions as sf
    required = [Path(sf.data_dir) / "MOUSE" / name for name in
                ("TRA.fasta", "TRB.fasta", "J-region-motifs.tsv", "C-region-motifs.tsv")]
    if not all(path.is_file() for path in required):
        pytest.skip("requires configured mouse Stitchr germlines")
    from DecoderTCR.reconstruct.components import prepare_gene_inputs
    from tcr_workbench.backends.mouse_stitch import stitch_mouse_tcrs
    from tcr_workbench.backends.reconstruct_worker import reconstruct

    genes = dict(trav="TRAV14-1", traj="TRAJ33", cdr3a="CAASDNYQLIW",
                 trbv="TRBV12-1", trbj="TRBJ2-7", cdr3b="CASSRANYEQYF",
                 peptide="ANYKFTLV")
    source, target = tmp_path / "input.csv", tmp_path / "output.csv"
    rows = [dict(genes, name="class-I", hla="H-2-Kb"),
            dict(genes, name="class-II", hla="H-2-IAb")]
    pl.DataFrame(rows).write_csv(source)
    reference = load_reference(biological_fingerprint("mouse")["mhc_reference_path"])["molecules"]
    raw = stitch_mouse_tcrs([dict(genes, name="mouse-golden")])["mouse-golden"]
    assert raw["TRA_leader"] == "TRAV14-1*01(L)"
    assert raw["TRB_leader"] == "TRBV12-1*01(L)"

    reconstruct(source, target, stitch=stitch_mouse_tcrs,
                lookup=lambda allele: (reference[allele]["HLA_a"], reference[allele]["HLA_b"]),
                convention="v2", species="mouse")
    output = pl.read_csv(target, infer_schema=False)
    public_row = dict(genes, name="mouse-golden", hla="H2-Kb")
    _, entries = prepare_gene_inputs([public_row], "v2", species="mouse")
    assert entries[0] is not None
    assert output["TCR_a"].to_list() == [entries[0]["sequences"]["TCR_a"]] * 2
    assert output["TCR_b"].to_list() == [entries[0]["sequences"]["TCR_b"]] * 2
    for index, allele in enumerate(("H-2-Kb", "H-2-IAb")):
        assert output["HLA_a"][index] == reference[allele]["HLA_a"]
        assert output["HLA_b"][index] == reference[allele]["HLA_b"]


@pytest.mark.parametrize("change", ["missing_chain", "invalid_amino_acid", "wrong_species", "wrong_class", "unknown_key", "empty_source"])
def test_user_reference_is_strict(tmp_path, change):
    reference = load_reference(biological_fingerprint("mouse")["mhc_reference_path"])
    molecule = reference["molecules"]["H-2-Kb"]
    if change == "missing_chain":
        del molecule["HLA_b"]
    elif change == "invalid_amino_acid":
        molecule["HLA_a"] += "X"
    elif change == "wrong_species":
        reference["species"] = "human"
    elif change == "wrong_class":
        molecule["class"] = "II"
    elif change == "unknown_key":
        reference["guess_partner"] = True
    else:
        molecule["source"] = " "
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError):
        biological_fingerprint("mouse", path)


def test_reference_hash_prevents_midrun_sequence_changes(tmp_path):
    source = Path(biological_fingerprint("mouse")["mhc_reference_path"])
    path = tmp_path / "reference.json"
    path.write_bytes(source.read_bytes())
    fingerprint = biological_fingerprint("mouse", path)
    assert fingerprint["mhc_reference_kind"] == "user-supplied"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        verify_biological_fingerprint(fingerprint)
    with pytest.raises(ValueError, match="mouse contexts only"):
        biological_fingerprint("human", path)
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_reference(path)


def test_mouse_ingestion_identity_and_export(tmp_path):
    path = tmp_path / "receptors.csv"
    pl.DataFrame([dict(**MOUSE_PAIR, cell_id="cell1", receptor_id="OT1", donor_id="mouse1")]).write_csv(path)
    mouse = read_receptors(path, species="mouse")
    human = read_receptors(path)
    assert mouse.species == "mouse"
    assert mouse.receptors["species"].to_list() == ["mouse"]
    assert mouse.receptors["receptor_id"][0] != human.receptors["receptor_id"][0]
    target = tmp_path / "pairs.csv"
    panel = pl.DataFrame({"peptide": ["SIINFEKL"], "hla": ["H-2-Kb"]})
    result = export_decoder_input(mouse.receptors, panel, target, species="mouse")
    assert result["exported_pairs"] == 1 and result["species"] == "mouse"
    assert _input_frame(target, species="mouse").height == 1
    with pytest.raises(ValueError, match="does not match"):
        _input_frame(target)
    pl.DataFrame([dict(**MOUSE_PAIR, species="human")]).write_csv(path)
    with pytest.raises(ValueError, match="species column disagrees"):
        read_receptors(path, species="mouse")


def test_mouse_pmhc_worker_uses_hashed_exact_reference_and_abstains(tmp_path):
    source, target = tmp_path / "input.csv", tmp_path / "reconstructed.csv"
    pl.DataFrame({"name": ["classI", "classII", "missing"], "hla": ["H-2-Kb", "H-2-IAb", "H-2-Kd"],
                  "peptide": ["SIINFEKL"] * 3}).write_csv(source)
    assert pmhc_decoder.validate_input(source, species="mouse").height == 3
    reference = biological_fingerprint("mouse")
    from tcr_workbench.backends import reconstruct_worker
    command = [sys.executable, str(Path(reconstruct_worker.__file__)), "--input", str(source), "--output", str(target),
               "--context-type", "pmhc", "--species", "mouse", "--mhc-reference", reference["mhc_reference_path"],
               "--mhc-reference-sha256", reference["mhc_reference_sha256"]]
    subprocess.run(command, check=True, capture_output=True, text=True)
    output = pl.read_csv(target)
    assert output["ok"].to_list() == [True, True, False]
    assert output["HLA_a"].str.len_chars().to_list()[:2] == [348, 233]
    assert "absent" in output["hla_reason"][2]
    command[-1] = hashlib.sha256(b"not the reference").hexdigest()
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode and "hash mismatch" in failed.stderr
