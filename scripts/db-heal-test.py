# Copyright (c) 2025 Sonic Operations Ltd
#
# Use of this software is governed by the Business Source License included
# in the LICENSE file and at soniclabs.com/bsl11.
#
# Change Date: 2028-4-16
#
# On the date above, in accordance with the Business Source License, use of
# this software will be governed by the GNU Lesser General Public License v3.

import argparse
import os
import subprocess
import tempfile
import time
import shutil
import sys
import re

# This script performs an experiment to recover database after a severe failure. It runs synchronisation of the 
# blockchain having the archive enabled. At a configured time, the process running the synchronisation is killed,
# simulating a failure. When this happens, the script reads what is the last checkpoint of the archive and recovers 
# the database to this checkpoint. Synchronisation continues from this checkpoint to verify that the database was 
# recovered successfully. 
#
# To stress test the process, the script runs in a loop with frequent checkpoint creation and a tight window to kill 
# the process. 
#
# The script needs python with default packages installed.  It needs Aida being build in a separate directory and
# further it needs the Aida database. These two directories must be configured.
#
# Example: python3 scripts/db-heal-test.py --aida /Path/To/Aida --aida-db /Path/To/Aida/Db

parser = argparse.ArgumentParser(prog="DB HEAL TEST SCRIPT",
                                 description="The script serves as a test tool for 'db-heal' feature."
                                             "It tests recover and LiveDB export/import.",
                                 usage="The script only works within Carmen root as it requires "
                                       "running some of the Carmen commands.\n"
                                       "To run the script, please provide Aida root using --aida and path to "
                                       "AidaDb using --aida-db.\n"
                                       "Example: python3 scripts/db-heal-test.py --aida /Path/To/Aida --aida-db"
                                       "/Path/To/Aida/Db")

# --- Parameters --- #
parser.add_argument('--aida', type=str, help="Path to Aida root.")
parser.add_argument('--aida-db', type=str, help="Path to AidaDB.")
parser.add_argument("--tmp", type=str, help="Path to tmp dir.")
parser.add_argument("--iter", type=int, help="Number of iterations.", default=1000)
parser.add_argument("--window", type=int,
                    help="Delay between start of sync process and forced termination (in seconds).", default=5)
parser.add_argument("--cp-granularity", type=int,
                    help="How often will Carmen create checkpoints (in blocks).", default=10)
parser.add_argument("--make", action=argparse.BooleanOptionalAction,
                    help="If enabled, necessary Aida tool will be build before the script begins.")

args = parser.parse_args()

aida_path = args.aida
aida_db_path = args.aida_db
tmp_path = args.tmp
number_of_iterations = args.iter
window = args.window
checkpoint_granularity = args.cp_granularity
make = args.make

# Find Carmen path within Aida directory
# (we always want to use same Carmen version for block processing and recovering)
carmen_path = os.path.join(aida_path, 'carmen', 'go')

# Mark first checkpoint
latest_checkpoint = checkpoint_granularity

if not aida_path or aida_path == "":
    print("please set Aida using --aida")
    sys.exit(1)
if not aida_db_path or aida_db_path == "":
    print("please set AidaDB using --aida-db")
    sys.exit(1)
if not tmp_path or tmp_path == "":
    tmp_path = tempfile.gettempdir()
    print(f"tmp not set - using default {tmp_path}")

# Block variables
first_block = 0
last_block = 60000000


# --- Script --- #

# Create working dir which gets deleted after the run
working_dir = tempfile.TemporaryDirectory(dir=tmp_path, prefix="db-heal-test_").name
if os.path.exists(working_dir):
    shutil.rmtree(working_dir)
os.makedirs(working_dir)

# Log file path from which we read output to find kill_block
aida_log_file = os.path.join(working_dir, 'aida.log')
carmen_log_file = os.path.join(working_dir, 'carmen.log')
genesis = os.path.join(working_dir, 'test_genesis.dat')

print("Your settings:")
print(f"\tNumber of iterations: {number_of_iterations}.")
print(f"\tSync time before kill: {window} seconds.")
print(f"\tCheckpoint granularity: {checkpoint_granularity} blocks.")
print(f"\tWorking directory:      {working_dir}")


# Function which checks programs return code, if program failed, log is printed and True is returned.
def has_program_failed(return_code, log, log_path):
    if return_code != 0:
        log.close()
        with open(log_path, 'r') as l:
            text = l.read()
            print(text)
        return True
    return False


# Function which checks every line added to aida_log_file and behaves accordingly to the line.
def check_aida_log(process, sleep_time, log_path):
    start = 0.0
    with open(log_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            if start > 0 and time.time() - start >= sleep_time:
                print("Interrupting...")
                process.terminate()
                return False
            if start == 0.0 and "Track: block" in line:
                # This line in combination with 0 start time means Syncing has been restarted.
                print("Syncing restarted!")
                start = time.time()
            # If process ends with error (return code 1) or either 'fail' or 'exit status' occurs in line exit script
            if process.poll() == 1 or any(s in line for s in ["exit status", "fail"]):
                print("Error occurred - printing output.log:")
                with open(aida_log_file, 'r') as l:
                    text = l.read()
                    print(text)
                return True


# Function which runs Carmen's info command and finds the latest checkpoint from created log
def get_latest_checkpoint_from_info(log_path):
    pattern = r"Checkpoint block:\s*(\d+)"
    cp = None
    checkpoint_keyword = 'Checkpoint: '
    with open(log_path, 'w') as cl:
        r = subprocess.run(
            ['go', 'run', './database/mpt/tool', 'info', str(archive)],
            stdout=cl,
            stderr=cl,
            cwd=carmen_path)
        if has_program_failed(r.returncode, cl, carmen_log_file):
            return -1

    with open(log_path, 'r') as cl:
        # Read the entire content of the file
        content = cl.read()
        # Search for the pattern in the file content
        match = re.search(pattern, content)
        if match:
            # Extract the number part from the match
            cp = match.group(1)
    return int(cp)


# Function which finds newest dir within path p hence finds the current working db.
def find_working_db(p):
    result = []
    for d in os.listdir(p):
        pd = os.path.join(p, d)
        if os.path.isdir(pd):
            result.append(pd)

    return max(result, key=os.path.getmtime)


if make:
    with open(aida_log_file, 'w') as f:
        print("Making Aida before run...")
        subprocess.run(
            ['make', 'aida-vm-sdb'],
            cwd=aida_path,
            stdout=f,
            stderr=subprocess.STDOUT
        )

# First iteration command
binary_path = os.path.join(aida_path, 'build', 'aida-vm-sdb')
cmd = [
    str(binary_path), 'substate', '--validate',
    '--db-tmp', working_dir, '--carmen-schema', '5', '--db-impl', 'carmen',
    '--aida-db', aida_db_path, '--no-heartbeat-logging', '--track-progress',
    '--archive', '--archive-variant', 's5', '--archive-query-rate', '200',
    '--carmen-checkpoint-interval', str(checkpoint_granularity), '--tracker-granularity',
    str(checkpoint_granularity), str(first_block), str(last_block)
]

with open(aida_log_file, 'w') as f:
    process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

print("Creating database with aida-vm-sdb...")

# Start monitoring the log file
has_failed = check_aida_log(process, window, aida_log_file)

# Wait for the first command to complete
process.wait()

if has_failed:
    sys.exit(1)

print("Testing db created, starting loop.")

for i in range(1, number_of_iterations + 1):

    # Find working dir - Aida copies db-src
    working_db = find_working_db(working_dir)
    archive = os.path.join(working_db, 'archive')
    live = os.path.join(working_db, 'live')

    # Dump carmen's logs into a file to avoid spamming
    c = open(carmen_log_file, 'w')

    # Find last checkpoint block
    latest_checkpoint = get_latest_checkpoint_from_info(carmen_log_file)
    print(f"Using checkpoint on block {latest_checkpoint}.")

    # Restore Archive
    result = subprocess.run(
        ['go', 'run', './database/mpt/tool', 'reset', '--force-unlock', str(archive), str(latest_checkpoint)],
        stdout=c,
        stderr=c,
        cwd=carmen_path)
    if has_program_failed(result.returncode, c, carmen_log_file):
        # Next error is fatal
        has_failed = True
        break

    # Export genesis to restore LiveDB
    print(f"Restoration complete. Exporting LiveDB genesis.")
    result = subprocess.run(
        ['go', 'run', './database/mpt/tool', 'export', '--block', str(latest_checkpoint), str(archive), str(genesis)],
        stdout=c,
        stderr=c,
        cwd=carmen_path)
    if has_program_failed(result.returncode, c, carmen_log_file):
        has_failed = True
        break

    # Restore LiveDB
    print("Export complete. Applying LiveDB genesis.")
    shutil.rmtree(live)

    result = subprocess.run(
        ['go', 'run', './database/mpt/tool', 'import-live-db', str(genesis), str(live)],
        stdout=c,
        stderr=c,
        cwd=carmen_path)
    if has_program_failed(result.returncode, c, carmen_log_file):
        has_failed = True
        break

    print(f"Iteration {i}/{number_of_iterations}")
    # We restored to block X, although we need to start the app at +1 block because X is already done
    first_block = latest_checkpoint + 1

    print("Restarting Aida...")
    command = [
        str(binary_path), 'substate', '--validate',
        '--db-tmp', working_dir, '--carmen-schema', '5', '--db-impl', 'carmen',
        '--aida-db', aida_db_path, '--no-heartbeat-logging', '--track-progress',
        '--archive', '--archive-variant', 's5', '--archive-query-rate', '200',
        '--carmen-checkpoint-interval', str(checkpoint_granularity), '--db-src',
        str(working_db), '--skip-priming', '--tracker-granularity',
        str(checkpoint_granularity), str(first_block), str(last_block)
    ]

    with open(aida_log_file, 'w') as f:
        process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)

    # Start monitoring the log file
    has_failed = check_aida_log(process, window, aida_log_file)

    # Wait for the command to complete
    process.wait()

    if has_failed:
        break

if has_failed:
    print("Fail")
    sys.exit(1)


# Clear anything leftover only if the script do not fail
print(f"Clearing work directory {working_dir}.")
shutil.rmtree(working_dir, ignore_errors=True)
print("Success!")
sys.exit(0)
