// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

/// Helper function to run a test with `shuttle` using the random scheduler for `_max_iter`
/// iterations.
/// It uses two environment variables to control its behavior:
/// - `SHUTTLE_REPLAY` to replay a run from the latest schedule file (if empty) or a specific one
///   (if set).
/// - `SHUTTLE_PERSISTENCE_MODE` to set the failure persistence mode. Values are:
///     - `file`: persist failures to a file.
///     - any other value or unset: print failures to stdout.
#[track_caller]
#[allow(dead_code)]
pub fn run_shuttle_check(_test: impl Fn() + Send + Sync + 'static, _num_iter: usize) {
    #[cfg(feature = "shuttle")]
    {
        if let Ok(schedule) = std::env::var("SHUTTLE_REPLAY") {
            let path = if !schedule.is_empty() {
                schedule
            } else {
                // Get the last created schedule file in the current directory
                let mut schedules = vec![];
                for entry in std::fs::read_dir(".").unwrap() {
                    let entry = entry.unwrap();
                    let path = entry.path();
                    if path.is_file()
                        && let Some(name) = path.file_name()
                        && name.to_string_lossy().starts_with("schedule")
                    {
                        schedules.push(path.to_string_lossy().to_string());
                    }
                }
                schedules.sort();
                schedules.last().cloned().expect("No schedule files found")
            };
            shuttle::replay_from_file(_test, &path);
        } else {
            let mut shuttle_config = shuttle::Config::new();
            shuttle_config.failure_persistence = match std::env::var("SHUTTLE_PERSISTENCE_MODE") {
                Ok(mode) if mode == "file" => shuttle::FailurePersistence::File(None),
                _ => shuttle::FailurePersistence::Print,
            };

            let runner = shuttle::Runner::new(
                shuttle::scheduler::RandomScheduler::new(_num_iter),
                shuttle_config,
            );
            runner.run(_test);
        }
    }
}

/// Helper function to set the name of the current shuttle task.
/// No-op if shuttle is not enabled.
#[allow(dead_code)]
pub fn set_name_for_shuttle_task(_name: String) {
    #[cfg(feature = "shuttle")]
    shuttle::current::set_name_for_task(shuttle::current::me(), _name);
}
