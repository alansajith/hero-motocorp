"""
Remap YOLO label files from old class scheme to new 3-class scheme.

Old mapping (from labeled dataset):
  0 → scratch
  1 → dent
  2 → crack
  3 → missing_part

New mapping:
  0 → scratch   (unchanged)
  1 → dent      (unchanged)
  2 → damage    (merged from crack=2 + missing_part=3)

Usage:
  python scripts/remap_labels.py --data-dir data
  python scripts/remap_labels.py --data-dir data --dry-run
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict


# Remapping: old_class_id -> new_class_id
CLASS_REMAP = {
    0: 0,  # scratch -> scratch
    1: 1,  # dent -> dent
    2: 2,  # crack -> damage
    3: 2,  # missing_part -> damage
}

NEW_CLASS_NAMES = {
    0: "scratch",
    1: "dent",
    2: "damage",
}

OLD_CLASS_NAMES = {
    0: "scratch",
    1: "dent",
    2: "crack",
    3: "missing_part",
}


def remap_label_file(label_path: Path, dry_run: bool = False) -> dict:
    """
    Remap class IDs in a single YOLO label file.

    Args:
        label_path: Path to the .txt label file
        dry_run: If True, don't modify the file

    Returns:
        Dict with stats: {'total_lines': int, 'remapped': int, 'changes': {old_id: count}}
    """
    stats = {
        "total_lines": 0,
        "remapped": 0,
        "changes": defaultdict(int),
        "skipped": 0,
    }

    lines = label_path.read_text().strip().split("\n")
    new_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        stats["total_lines"] += 1
        parts = line.split()

        if len(parts) < 5:
            # Invalid line format, keep as-is
            new_lines.append(line)
            stats["skipped"] += 1
            continue

        old_class_id = int(parts[0])

        if old_class_id not in CLASS_REMAP:
            print(f"  ⚠️  Unknown class ID {old_class_id} in {label_path.name}, keeping as-is")
            new_lines.append(line)
            stats["skipped"] += 1
            continue

        new_class_id = CLASS_REMAP[old_class_id]

        if old_class_id != new_class_id:
            stats["remapped"] += 1
            stats["changes"][old_class_id] += 1

        parts[0] = str(new_class_id)
        new_lines.append(" ".join(parts))

    if not dry_run:
        label_path.write_text("\n".join(new_lines) + "\n")

    return stats


def remap_dataset(data_dir: Path, dry_run: bool = False, backup: bool = True):
    """
    Remap all label files in the dataset.

    Args:
        data_dir: Root data directory containing train/val/test splits
        dry_run: If True, only report what would change
        backup: If True, backup original labels before remapping
    """
    splits = ["train", "val", "test"]
    total_stats = {
        "files_processed": 0,
        "files_modified": 0,
        "total_annotations": 0,
        "total_remapped": 0,
        "class_changes": defaultdict(int),
        "class_distribution_before": defaultdict(int),
        "class_distribution_after": defaultdict(int),
    }

    print("=" * 60)
    print("YOLO Label Class Remapping")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will modify files)'}")
    print(f"\nRemapping rules:")
    for old_id, new_id in CLASS_REMAP.items():
        old_name = OLD_CLASS_NAMES.get(old_id, f"class_{old_id}")
        new_name = NEW_CLASS_NAMES.get(new_id, f"class_{new_id}")
        arrow = "→" if old_id != new_id else "="
        print(f"  {old_id} ({old_name}) {arrow} {new_id} ({new_name})")
    print()

    for split in splits:
        labels_dir = data_dir / split / "labels"

        if not labels_dir.exists():
            print(f"⚠️  {split}/labels not found, skipping...")
            continue

        print(f"\n{'─' * 40}")
        print(f"Processing: {split}/labels")
        print(f"{'─' * 40}")

        # Backup original labels
        if backup and not dry_run:
            backup_dir = data_dir / split / "labels_backup_original"
            if not backup_dir.exists():
                print(f"  📦 Backing up to {backup_dir.name}/")
                shutil.copytree(labels_dir, backup_dir)
            else:
                print(f"  📦 Backup already exists at {backup_dir.name}/")

        label_files = sorted(labels_dir.glob("*.txt"))
        print(f"  Found {len(label_files)} label files")

        split_remapped = 0

        for label_file in label_files:
            # Count class distribution before
            lines = label_file.read_text().strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and len(line.split()) >= 5:
                    class_id = int(line.split()[0])
                    total_stats["class_distribution_before"][class_id] += 1

            stats = remap_label_file(label_file, dry_run=dry_run)

            total_stats["files_processed"] += 1
            total_stats["total_annotations"] += stats["total_lines"]
            total_stats["total_remapped"] += stats["remapped"]

            if stats["remapped"] > 0:
                total_stats["files_modified"] += 1
                split_remapped += 1

            for old_id, count in stats["changes"].items():
                total_stats["class_changes"][old_id] += count

        print(f"  ✅ {split_remapped} files had remapped annotations")

    # Count class distribution after
    if not dry_run:
        for split in splits:
            labels_dir = data_dir / split / "labels"
            if not labels_dir.exists():
                continue
            for label_file in sorted(labels_dir.glob("*.txt")):
                lines = label_file.read_text().strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line and len(line.split()) >= 5:
                        class_id = int(line.split()[0])
                        total_stats["class_distribution_after"][class_id] += 1
    else:
        # Simulate after distribution
        for class_id, count in total_stats["class_distribution_before"].items():
            new_id = CLASS_REMAP.get(class_id, class_id)
            total_stats["class_distribution_after"][new_id] += count

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Files processed:  {total_stats['files_processed']}")
    print(f"  Files modified:   {total_stats['files_modified']}")
    print(f"  Total annotations: {total_stats['total_annotations']}")
    print(f"  Annotations remapped: {total_stats['total_remapped']}")

    if total_stats["class_changes"]:
        print(f"\n  Class changes:")
        for old_id, count in sorted(total_stats["class_changes"].items()):
            old_name = OLD_CLASS_NAMES.get(old_id, f"class_{old_id}")
            new_name = NEW_CLASS_NAMES.get(CLASS_REMAP[old_id], f"class_{CLASS_REMAP[old_id]}")
            print(f"    {old_name} (id={old_id}) → {new_name} (id={CLASS_REMAP[old_id]}): {count} annotations")

    print(f"\n  Class distribution BEFORE:")
    for class_id in sorted(total_stats["class_distribution_before"].keys()):
        name = OLD_CLASS_NAMES.get(class_id, f"class_{class_id}")
        count = total_stats["class_distribution_before"][class_id]
        print(f"    {class_id}: {name:15s} → {count:6d} annotations")

    print(f"\n  Class distribution AFTER:")
    for class_id in sorted(total_stats["class_distribution_after"].keys()):
        name = NEW_CLASS_NAMES.get(class_id, f"class_{class_id}")
        count = total_stats["class_distribution_after"][class_id]
        print(f"    {class_id}: {name:15s} → {count:6d} annotations")

    if dry_run:
        print(f"\n  💡 This was a DRY RUN. No files were modified.")
        print(f"     Run without --dry-run to apply changes.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Remap YOLO label class IDs for vehicle damage detection"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root data directory (contains train/val/test subdirs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would change, don't modify files",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up original label files",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    remap_dataset(data_dir, dry_run=args.dry_run, backup=not args.no_backup)


if __name__ == "__main__":
    main()
