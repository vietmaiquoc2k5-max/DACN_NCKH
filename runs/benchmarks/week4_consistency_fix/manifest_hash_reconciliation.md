# Manifest Hash Reconciliation

Current worktree raw-byte SHA-256: `5ba1855d624f65339f60fcd741ed2edc874ac5e327529a0d2c99c8460592ef23`.
Current UTF-8 text after universal-newline normalization SHA-256: `3afc88d254ab3eb7284ecfd304b9c4cb75ace1a8f55bf946eb1f108d2533bb40`.
Current Git HEAD blob SHA-256: `f7112140f651bf9740a3fb654770590748192727`.
Historical inference run recorded: `75626b6b563bae499848543732cfb1a172e1475773674b1acbd36a80e8e0c1ed`.
Historical inference reconciliation recorded: `3afc88d254ab3eb7284ecfd304b9c4cb75ace1a8f55bf946eb1f108d2533bb40`.
Historical provenance audit recorded: `5ba1855d624f65339f60fcd741ed2edc874ac5e327529a0d2c99c8460592ef23`.

The 5413… raw hash corresponds to the CRLF rendering of the prior 2.3 manifest at commit 3ebd; 4bf… is its LF Git blob. The current v2.3 file's raw CRLF hash is 5ba… and its LF/Git blob hash is 3af…. These are line-ending representations of their respective committed contents, not interchangeable hash inputs.
The inference run's 756… digest is not reproduced by current bytes or current normalized text. Its exact source manifest snapshot is not present in this repository; classify it as HISTORICAL_SNAPSHOT / UNVERIFIED, not a match.
