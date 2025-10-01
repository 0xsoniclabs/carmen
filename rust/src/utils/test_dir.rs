// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

use std::{
    fs::{self},
    ops::Deref,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

/// An enum representing different UNIX file permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permissions {
    ReadWrite,
    ReadOnly,
    WriteOnly,
}

impl From<Permissions> for std::fs::Permissions {
    fn from(permission: Permissions) -> Self {
        std::fs::Permissions::from_mode(permission.mode())
    }
}

impl Permissions {
    /// Returns the UNIX mode for the given permissions.
    /// Note that the execute bit is always set.
    pub fn mode(self) -> u32 {
        match self {
            Permissions::ReadWrite => 0o777,
            Permissions::ReadOnly => 0o555,
            Permissions::WriteOnly => 0o333,
        }
    }
}

/// A utility struct for creating and managing temporary test directories with specific permissions.
/// On drop, it will delete the directory and its contents.
pub struct TestDir {
    dir: PathBuf,
}

impl TestDir {
    /// Creates a new temporary `TestDir` with the specified permissions.
    pub fn try_new(permission: Permissions) -> std::io::Result<Self> {
        let dir = tempfile::tempdir()?;
        set_permissions(dir.path(), permission)?;
        Ok(Self { dir: dir.keep() })
    }

    /// Returns the path of the test directory.
    pub fn path(&self) -> &std::path::Path {
        &self.dir
    }

    /// Recursively sets the permissions of the test directory and its contents.
    pub fn set_permissions(&self, permission: Permissions) -> std::io::Result<()> {
        set_permissions(&self.dir, permission)
    }
}

impl AsRef<Path> for TestDir {
    fn as_ref(&self) -> &Path {
        &self.dir
    }
}

impl Deref for TestDir {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        self.path()
    }
}

impl Drop for TestDir {
    /// Deletes the test directory and its contents
    fn drop(&mut self) {
        set_permissions(&self.dir, Permissions::ReadWrite).unwrap_or_else(|e| {
            eprintln!(
                "Failed to set permissions for test directory {}: {}",
                self.dir.display(),
                e
            );
        });
        fs::remove_dir_all(&self.dir).unwrap_or_else(|e| {
            eprintln!(
                "Failed to remove test directory {}: {}",
                self.dir.display(),
                e
            );
        });
    }
}

/// Recursively set permissions for a directory and its contents
pub fn set_permissions(dir: &Path, permission: Permissions) -> std::io::Result<()> {
    // First make root directory readable so we can list its contents.
    fs::set_permissions(dir, Permissions::ReadOnly.into())?;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            set_permissions(&entry.path(), permission)?;
        } else {
            fs::set_permissions(entry.path(), permission.into())?;
        }
    }
    // Set permissions for the root directory
    fs::set_permissions(dir, permission.into())?;
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn permissions_mode_returns_correct_mode() {
        assert_eq!(Permissions::ReadWrite.mode(), 0o777);
        assert_eq!(Permissions::ReadOnly.mode(), 0o555);
        assert_eq!(Permissions::WriteOnly.mode(), 0o333);
    }

    #[test]
    fn permissions_from_std_fs_permissions_returns_correct_permissions() {
        assert_eq!(
            std::fs::Permissions::from_mode(0o777),
            Permissions::ReadWrite.into()
        );
        assert_eq!(
            std::fs::Permissions::from_mode(0o555),
            Permissions::ReadOnly.into()
        );
        assert_eq!(
            std::fs::Permissions::from_mode(0o333),
            Permissions::WriteOnly.into()
        );
    }

    #[test]
    fn set_permissions_sets_permissions_recursively() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        fs::create_dir(test_dir.join("subdir")).unwrap();

        test_dir.set_permissions(Permissions::WriteOnly).unwrap();

        let permissions = fs::metadata(&test_dir).unwrap().permissions();
        assert_eq!(permissions.mode() & 0o777, Permissions::WriteOnly.mode());
        let subdir_permissions = fs::metadata(test_dir.join("subdir")).unwrap().permissions();
        assert_eq!(
            subdir_permissions.mode() & 0o777,
            Permissions::WriteOnly.mode()
        );
    }

    #[test]
    fn set_permissions_sets_permissions_of_write_only_directory() {
        let test_dir = TestDir::try_new(Permissions::WriteOnly).unwrap();
        fs::create_dir(test_dir.join("subdir")).unwrap();

        test_dir.set_permissions(Permissions::ReadOnly).unwrap();

        let permissions = fs::metadata(&test_dir).unwrap().permissions();
        assert_eq!(permissions.mode() & 0o777, Permissions::ReadOnly.mode());
        let subdir_permissions = fs::metadata(test_dir.join("subdir")).unwrap().permissions();
        assert_eq!(
            subdir_permissions.mode() & 0o777,
            Permissions::ReadOnly.mode()
        );
    }

    #[test]
    fn set_permissions_fails_for_non_existent_directory() {
        let non_existent_path = PathBuf::from("non_existent_dir");
        let result = set_permissions(&non_existent_path, Permissions::ReadWrite)
            .expect_err("set_permissions should fail for non-existent directory");
        assert_eq!(result.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn path_returns_path() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        assert!(test_dir.exists());
    }

    #[test]
    fn make_read_only_changes_permissions() {
        let test_dir = TestDir::try_new(Permissions::ReadOnly).unwrap();
        let permissions = fs::metadata(test_dir).unwrap().permissions();
        assert_eq!(permissions.mode() & 0o777, Permissions::ReadOnly.mode());
    }

    #[test]
    fn make_write_only_changes_permissions() {
        let test_dir = TestDir::try_new(Permissions::WriteOnly).unwrap();
        let permissions = fs::metadata(test_dir).unwrap().permissions();
        assert_eq!(permissions.mode() & 0o777, Permissions::WriteOnly.mode());
    }

    #[test]
    fn make_read_write_changes_permissions() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        let permissions = fs::metadata(test_dir).unwrap().permissions();
        assert_eq!(permissions.mode() & 0o777, Permissions::ReadWrite.mode());
    }

    #[test]
    fn as_ref_returns_path() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        assert_eq!(test_dir.as_ref(), test_dir.path());
    }

    #[test]
    fn deref_returns_path() {
        let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
        assert_eq!(test_dir.deref(), test_dir.path());
    }

    #[test]
    fn drop_deletes_directory_and_subdirectories() {
        let init_dir = |permission: Permissions| {
            // Create it as read write to be able to create subdirectories
            let test_dir = TestDir::try_new(Permissions::ReadWrite).unwrap();
            assert!(test_dir.exists());
            fs::create_dir(test_dir.join("subdir")).unwrap();
            // Apply the specified permissions
            set_permissions(&test_dir, permission).unwrap();
            test_dir
        };

        // Read-only permissions
        let test_dir = init_dir(Permissions::ReadOnly);
        let path = test_dir.to_path_buf();
        drop(test_dir);
        assert!(!path.exists());

        // Write-only permissions
        let test_dir = init_dir(Permissions::WriteOnly);
        let path = test_dir.to_path_buf();
        drop(test_dir);
        assert!(!path.exists());

        // Read-write permissions
        let test_dir = init_dir(Permissions::ReadWrite);
        let path = test_dir.to_path_buf();
        drop(test_dir);
        assert!(!path.exists());
    }
}
