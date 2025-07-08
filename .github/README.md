# GitHub Actions CI/CD

This directory contains GitHub Actions workflows for continuous integration and automated releases.

## Workflows

### CI (`ci.yml`)
Runs on every push to main and on all pull requests.

- **Tests**: Runs tests on Ubuntu and macOS using Hermit-managed Zig
- **Format Check**: Ensures code is properly formatted with `zig fmt`
- **Build**: Builds in both debug and release modes
- **Benchmarks**: Runs performance benchmarks on Linux

### Auto Release (`release.yml`)
Automatically creates releases based on conventional commits when pushing to main.

- Uses `svu` (Semantic Version Utility) to analyze commits and determine version bumps:
  - `feat:` commits trigger a minor version bump (0.x.0)
  - `fix:` commits trigger a patch version bump (0.0.x)
  - Breaking changes (`feat!:` or `BREAKING CHANGE`) trigger a major version bump (x.0.0)
- Updates version in `build.zig.zon`
- Creates GitHub release with auto-generated release notes

### Manual Release (`manual-release.yml`)
Allows manual triggering of releases via GitHub's workflow dispatch.

- Specify exact version number (e.g., 1.2.3)
- Validates semver format
- Creates release with specified version

## Conventional Commits

This project follows [Conventional Commits](https://www.conventionalcommits.org/) for automatic versioning:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Common types:
- `feat`: New feature (triggers minor version bump)
- `fix`: Bug fix (triggers patch version bump)
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or fixes
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Breaking Changes:
Add `!` after the type or include `BREAKING CHANGE:` in the footer to trigger a major version bump:

```
feat!: remove deprecated API

BREAKING CHANGE: The old API has been removed in favor of the new one.
```

## Version Management

Versions are stored in `build.zig.zon` and follow semantic versioning (MAJOR.MINOR.PATCH).

The auto-release workflow uses `svu` to:
1. Analyze commits since the last tag
2. Determine the appropriate version bump based on conventional commits
3. Update `build.zig.zon` with the new version
4. Create a git tag and GitHub release

### Tools Used

- **Hermit**: Package manager for consistent tool versions across environments
- **svu**: Semantic Version Utility for analyzing commits and calculating versions
- **Zig**: Managed via Hermit to ensure consistent version (0.14.0)

## Security

- Workflows have minimal permissions (only `contents: write` for releases)
- Uses pinned action versions for security
- Bot commits are clearly marked and use `[skip ci]` to prevent loops
