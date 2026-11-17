"""Identity-bound exception for the unchanged PILOT ImageNet-R preparation.

The public preparation contains 23 byte-identical train/test pairs. Keeping the
published split is an explicit protocol choice, never a generic overlap bypass.
Split identities below hash the complete ordered ImageFolder manifest, including
every relative image path and its file SHA256, using compact JSON (UTF-8).
"""

from collections import Counter
import hashlib
import json


MANIFEST_SCHEME = "imagefolder_relative_path_file_sha256_compact_json_v1"
SOURCE_URL = (
    "https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/"
    "EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW&download=1"
)
SOURCE_ARCHIVE_SHA256 = "c2720edb18da7173b6cb6d9e9a65e18fb6f949f65d8cd10ac99df9003d8b055a"
SPLIT_SHA256 = {
    "train": "925706d3908e5ebe55c5a18565f5699786074013cee3f1fcd525095836b19035",
    "test": "ff2d41d250c2223126e3749d0b9eaed8d926e1f5f06725a11b4a8a312652eba2",
}
SPLIT_COUNTS = {"train": 24000, "test": 6000}
KNOWN_OVERLAP_SHA256 = frozenset({
    "02fe5297e4765070d458fe8ca6520d74e42c8e315ad6f7b4256f45677b7208ab",
    "0f99820fe0cf33887f96b588a8bf79924bd471be462b47ffd99bf0d4eadaaa18",
    "2b6c78e60e2dd59f323823fa98ed8d6d24aab2e6b8428bfe1df8326f5252a365",
    "2bc576eb0e8a9775ef37a1cccf00058f54dd6ea54671e129c8081f406f9921bc",
    "3a0c802a05d7f5173975c1fe492bb150eab82ba61c898c48c1cfe9f219ec5cda",
    "421177dea3742b09471dda087ecbbecbd3692a63856b9169c0a9618490011f08",
    "4839327c8a7f170a8d48875e10490b701e52f7957dc0f7279bb02d8e04106f63",
    "4ea2727c7715e7baa70db2754c59d564036d20d2156c1bc9cca77f12899a487c",
    "587c395627bbf0b3a40e931b931b293ff17a27162be6f1fd02df5ee0d62ad545",
    "5c104dce25df24aa1021928feccf8b412a0b70ba5f3fc968d61467c8faa4b72e",
    "6645c2649ad102911118fcf73c67979e318a2201f7d54d564a76cd1129a327a1",
    "6b07a63fdc550282fe4d747bcce153d83a6bc5ca99cf7ba525c86fd220914dba",
    "6c5c97e86b88bd2c12f8652bf941b2a29a7f6db3a5918d7b0a166efa0d17f31c",
    "8d03626caa461872a784dae4d89e88041cecc29b61ea4ffe7c734c5719c6db2a",
    "b06b9feefd7db55e1cccd687b5c1b3cdcfacce3d78bf7da5ff807087f524cef9",
    "b0c7fe2907c5b208f4497187482d8d13075eafc071bcfbfd59bd0f26f3574cb0",
    "bd4dde8dea970e76df3fce9ec0f498b4a4a5585a7c6ecdd53d80866aabf19a7c",
    "bdc10c823561eedbf7bdceccd52e566b5b6463729a302c8593acc98a6312e209",
    "c15a66290c548aa79230597ed075a0934c199795f685c76328c39e3c4ead4cc6",
    "cb05f11d167624db884078731d305aa72f19f06d0e1f97f4eec9a956d3a11095",
    "cd2d573ecaf8230f978653c5341f45917aed2160b49d763c0d5378898a809824",
    "ce20765dc495bb5e9445d8dcd289c4631d89e0a0243fbde8bba0f68029072579",
    "fb0c23d0c655a5f14758921714327b6dab26240e4d7792019b6c5be8309df70a",
})


def validate_prepared_split(policy, split_digests, inventory, development):
    """Check actual tree identities before allowing the known original overlap.

    User-supplied archive hashes are deliberately not accepted as evidence.
    Development checks only train and never requests or infers test contents.
    """
    if policy not in {"reject", "preserve_verified_pilot"}:
        raise ValueError("Unknown prepared_split_overlap_policy")
    if policy == "preserve_verified_pilot":
        checked = ["train"] if development else ["train", "test"]
        for split in checked:
            if (
                split_digests.get(split) != SPLIT_SHA256[split]
                or len(inventory.get(split, [])) != SPLIT_COUNTS[split]
            ):
                raise ValueError(
                    f"Prepared PILOT {split} tree differs from the verified public preparation"
                )
    overlap = None
    pairs = None
    if not development:
        train = Counter(inventory["train"])
        test = Counter(inventory["test"])
        overlap = set(train) & set(test)
        pairs = sum(train[digest] * test[digest] for digest in overlap)
        if policy == "reject" and overlap:
            raise ValueError(
                "Prepared ImageNet-R train/test splits contain identical image files"
            )
        if policy == "preserve_verified_pilot" and (
            overlap != KNOWN_OVERLAP_SHA256
            or any(train[h] != 1 or test[h] != 1 for h in overlap)
        ):
            raise ValueError("Prepared PILOT overlap differs from the verified public preparation")
    report = {
        "policy": policy,
        "manifest_scheme": MANIFEST_SCHEME,
        "verified_splits": (
            (["train"] if development else ["train", "test"])
            if policy == "preserve_verified_pilot" else []
        ),
        "observed_split_manifest_sha256": dict(split_digests),
        "observed_cross_split_duplicate_hash_count": (
            len(overlap) if overlap is not None else None
        ),
        "observed_cross_split_duplicate_pair_count": pairs,
        "original_split_preserved": True,
        "deduplicated": False,
    }
    if policy == "preserve_verified_pilot":
        report.update({
            "source_url": SOURCE_URL,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "identity_evidence": "Actual extracted-tree manifests matched; archive SHA is reference provenance",
            "reference_split_manifest_sha256": dict(SPLIT_SHA256),
            "known_cross_split_duplicate_hash_count": len(KNOWN_OVERLAP_SHA256),
            "known_overlap_hashes_sha256": hashlib.sha256(
                json.dumps(sorted(KNOWN_OVERLAP_SHA256), separators=(",", ":")).encode()
            ).hexdigest(),
            "known_overlap_definition": "cl27/pilot_imagenetr.py:KNOWN_OVERLAP_SHA256",
        })
    return report
