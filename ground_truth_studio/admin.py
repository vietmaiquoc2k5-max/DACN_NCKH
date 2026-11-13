"""Explicit local operator provisioning; no default account, signoff or annotation."""
from __future__ import annotations

import argparse
import getpass

from ground_truth_studio.config import Settings
from ground_truth_studio.review_repository import ReviewRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    account = actions.add_parser("create-account")
    account.add_argument("--reviewer-id", required=True)
    account.add_argument("--person-id", required=True, help="Stable ID verified by the human operator; do not create aliases for one person")
    account.add_argument("--role", required=True, choices=["annotator_01", "annotator_02", "adjudicator", "admin"])
    account.add_argument("--identity-evidence", required=True)
    refresh = actions.add_parser("snapshot-csv")
    refresh.add_argument("--accept-external-changes", action="store_true")
    arguments = parser.parse_args()
    repository = ReviewRepository(Settings.load())
    if arguments.action == "create-account":
        password = getpass.getpass("Mật khẩu (>=12 ký tự): ")
        if password != getpass.getpass("Nhập lại mật khẩu: "):
            parser.error("Passwords differ")
        repository.provision(arguments.reviewer_id, arguments.person_id, arguments.role, password, arguments.identity_evidence)
        print("Đã tạo tài khoản. Không có human signoff/annotation nào được tạo.")
    else:
        conflicts = repository.snapshot_csv(accept_external_changes=arguments.accept_external_changes)
        print("CSV conflicts:", conflicts)


if __name__ == "__main__":
    main()
