# Releasing

Releases are built and validated in GitHub Actions, published manually to
TestPyPI, and published to PyPI only from a GitHub Release. Both uploads use
short-lived OIDC credentials and produce package attestations; no API token is
stored in GitHub.

## One-time trusted publishing setup

Create a `testpypi` GitHub environment and a `pypi` environment. Configure the
`pypi` environment with required reviewers so production releases need explicit
approval.

Register pending trusted publishers in both
[PyPI](https://pypi.org/manage/account/publishing/) and
[TestPyPI](https://test.pypi.org/manage/account/publishing/) with these values:

| Field | Value |
| --- | --- |
| PyPI project name | `omni-forecast` |
| GitHub owner | `hbmartin` |
| GitHub repository | `omni-forecast` |
| Workflow | `publish.yml` |
| Environment | `pypi` on PyPI; `testpypi` on TestPyPI |

TestPyPI uses a separate account from PyPI. The first successful trusted upload
creates the project when a pending publisher is configured.

## Release checklist

1. Choose a version that has never been uploaded to the target index. PyPI
   artifacts are immutable. Update the project and lockfile together:

   ```bash
   uv version 0.1.0
   uv lock
   ```

2. Run the complete local gate:

   ```bash
   uv run ruff check src --fix
   uv run ruff format src tests
   uv run pyrefly check src
   uv run ty check src
   uv run deptry src
   uv run pyroma --min 8 .
   uv run lizard -Eduplicate -C 27 src
   uv run pytest tests/ --cov=src --cov-report=term-missing
   uv build --no-sources
   uvx --from twine==6.2.0 twine check dist/*
   ```

3. Merge the release commit into `main` and confirm the CI and Docs workflows
   pass. Verify that <https://hbmartin.github.io/omni-forecast/> is live before
   publishing because it is included in the package metadata.

4. Manually run the **Publish** workflow against that commit to upload it to
   TestPyPI. Use a unique prerelease version such as `0.1.0rc1` if more than one
   TestPyPI rehearsal may be needed.

5. Verify the TestPyPI project page and the workflow's installed-wheel smoke
   test. Promote the tested code to the final version if it used a prerelease.

6. Tag the final commit `v<version>` and create a GitHub Release from that tag.
   The workflow refuses to publish when the release tag does not exactly match
   the version in `pyproject.toml`.

7. Approve the `pypi` environment deployment. After publication, verify the
   project page and a clean installation from PyPI.
