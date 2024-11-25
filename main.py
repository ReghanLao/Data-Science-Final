#!/usr/bin/env python3
import subprocess
import os
import sys

def run_hadoop_streaming(input_file):
    # Define a fixed output directory name
    output_dir = "output"

    # Delete existing output directory if it exists
    if os.path.exists(output_dir):
        os.system(f"rm -r {output_dir}")

    # Define local paths to your mapper and reducer
    mapper_script = "mapper.py"
    reducer_script = "reducer.py"

    # Build the Hadoop streaming command
    command = [
        "hadoop", "jar",
        "/common/system/hadoop-3.4.0/share/hadoop/tools/lib/hadoop-streaming-3.4.0.jar",
        "-D", "mapreduce.local.dir=/freespace/local/jsb337",
        "-D", "mapreduce.cluster.local.dir=/freespace/local/jsb337",
        "-D", "mapreduce.job.local.dir=/freespace/local/jsb337",
        "-D", "mapreduce.local.map.tasks.maximum=8",
        "-D", "mapreduce.job.reduces=1",
        "-D", "mapreduce.map.memory.mb=8192",
        "-D", "mapreduce.map.java.opts=-Xmx6144m",
        "-D", "mapreduce.reduce.memory.mb=16384",
        "-D", "mapreduce.reduce.java.opts=-Xmx12288m",
        "-file", mapper_script,
        "-file", reducer_script,
        "-mapper", f"python3 {mapper_script}",
        "-reducer", f"python3 {reducer_script}",
        "-input", input_file,
        "-output", output_dir
    ]

    # Run the Hadoop streaming job
    subprocess.run(command, check=True)

if __name__ == "__main__":
    # Check for input argument
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    run_hadoop_streaming(input_file)