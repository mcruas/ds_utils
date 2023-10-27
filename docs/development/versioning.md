This document shows how to create a new version of this project. The objective is to remind the author the necessary steps. It is not a complete guide.

# How to create a new version

## Project files

- `README.md`: If you've added or removed features, or if you've changed the package's description.


## `Setup.py` file

If you're creating a new version for your package, it's recommended to update the `setup.py` file to reflect this new version number. Specifically, you should update the `version` argument in the `setup()` function.

Here's what you need to do:

### 1. Update the `version` Argument:

In your `setup.py` file, find the `version` argument and update it:

Before:
```python
setup(
    ...
    version='0.0.1',  # Or whatever your initial version was.
    ...
)
```

After:
```python
setup(
    ...
    version='0.1',
    ...
)
```

### 2. Optional but Recommended: Update Other Metadata

While you're at it, consider updating other metadata that may have changed:

- `description`: If the nature or primary features of your package have changed.
- `install_requires`: If there are any new dependencies.
- `url`: If you've moved your project to a new URL or if you initially left this out.
- `classifiers`: To provide additional metadata about the version (e.g., Development Status, Environment, etc.)

### 3. Commit the Changes:

After making the changes to `setup.py`, you should commit them:

```bash
git add setup.py
git commit -m "Update setup.py for 0.1 release"
```

Then, you can proceed with the version tagging and pushing to GitHub as previously described.

### 4. Future Versioning:

For future versions, it's good practice to always update the `version` in your `setup.py` as part of your versioning workflow. This ensures consistency between the tagged version in your Git repository and the version metadata in your package. It also helps if you or others want to distribute the package via PyPI or other platforms, as the version in `setup.py` will be used as the package's version.


## Github

Creating a version `0.1` of your package and updating it on GitHub requires a series of steps involving both Git and GitHub. Here's a step-by-step guide:

### 1. Ensure All Changes Are Committed:

Before you tag a version, ensure all your changes are committed to your local Git repository.

Navigate to your project's directory:

```bash
cd path/to/your/ds_utils_project
```

Check for uncommitted changes:

```bash
git status
```

If there are any changes, add and commit them:

```bash
git add .
git commit -m "Prepare for 0.1 release"
```

### 2. Tagging the Version:

Tag the current commit as `v0.1`:

```bash
git tag -a v0.1 -m "Version 0.1 release"
```

This command creates an annotated tag. The `-a` flag allows you to add a message (`-m`) with the tag, describing the release or any relevant information.

### 3. Pushing Changes and Tags to GitHub:

Push your changes (if any) to GitHub:

```bash
git push origin main  # Assuming you're using the 'main' branch. Replace with 'master' or another branch name if needed.
```

Push your tags:

```bash
git push origin v0.1
```

### 4. Creating a Release on GitHub:

Now, to create a more detailed release on GitHub (optional but recommended):

1. Go to your GitHub repository in a web browser.
2. Click on the "Releases" tab.
3. Click "Draft a new release" or "Create a new release."
4. In the "Tag version" dropdown, select the tag `v0.1` you just pushed.
5. Give your release a title (e.g., "Initial Release") and describe the changes or features included in this version.
6. If necessary, attach any binaries or assets.
7. Click "Publish release."

### 5. Celebrate!

You've just versioned your package and updated it on GitHub. As you continue to develop your package, you can increment the version numbers (following [semantic versioning](https://semver.org/) practices) and repeat the above process for future releases.

Remember that having clear and informative commit and tag messages helps both you and others understand the history and progression of your project.