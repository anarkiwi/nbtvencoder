# Releasing

Releases are published to [PyPI](https://pypi.org/project/nbtvencoder/) by the
[`Release`](.github/workflows/release.yml) workflow using **PyPI Trusted
Publishing** (OpenID Connect). No API tokens or passwords are stored anywhere —
GitHub mints a short-lived OIDC token that PyPI verifies against the trusted
publisher you register below.

## One-time setup (PyPI trusted publisher)

`nbtvencoder` is not on PyPI yet, so register a **pending publisher**, which both
authorises the workflow and reserves the project name:

1. Sign in to PyPI and open
   <https://pypi.org/manage/account/publishing/>.
2. Under **Add a new pending publisher → GitHub**, enter exactly:

   | Field | Value |
   | --- | --- |
   | PyPI Project Name | `nbtvencoder` |
   | Owner | `anarkiwi` |
   | Repository name | `nbtvencoder` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

3. Click **Add**.

These values match `.github/workflows/release.yml` (the workflow file name and
its `environment: pypi`). Once the project's first release is published, the
pending publisher becomes a normal trusted publisher; manage it later at
*Project → Settings → Publishing*.

> Optional hardening: in the GitHub repo, open *Settings → Environments → pypi*
> and add protection rules (e.g. required reviewers, or restrict deployments to
> tags matching `v*`). The `pypi` environment already exists.

## Cutting a release

1. Bump the version in [`src/nbtvencoder/_version.py`](src/nbtvencoder/_version.py)
   (e.g. `0.1.0` → `0.1.1`) and update [`CHANGELOG.md`](CHANGELOG.md).
2. Open a PR, let CI pass, and merge to `main`.
3. Create a **GitHub Release** with a tag of `v<version>` (for example `v0.1.1`)
   targeting `main`. The tag must match the package version — the workflow fails
   the build otherwise.
4. Publishing the release triggers `release.yml`, which builds the sdist + wheel,
   runs `twine check`, and uploads to PyPI via the trusted publisher.

You can watch it with `gh run watch` or under the repository's **Actions** tab.

## Testing the publish flow (optional)

To rehearse against [TestPyPI](https://test.pypi.org/) first, register an
equivalent pending publisher there and add a job that points
`pypa/gh-action-pypi-publish` at `https://test.pypi.org/legacy/` via its
`repository-url` input.
