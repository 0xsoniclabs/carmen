import sys
import os
import re
from collections.abc import Iterable


def apply_in_context(
    input_lines, output_lines, end_context_delimiter, matching, transform,
):
    first_match = False
    while input_lines[0].strip() != end_context_delimiter:
        line = input_lines[0]
        if matching(line.strip()):
            first_match = True
            output_lines.append(transform(input_lines))
        else:
            def get_line(input_lines):
                line = input_lines.pop(0)
                output_lines.append(line)
                return line
            if first_match:
                traverse_line_scoped(input_lines, get_line)
            else:
                output_lines.append(input_lines.pop(0))
    output_lines.append(input_lines.pop(0))


def comment_out_line(line: str):
    return "// " + line

def traverse_line_scoped(input_lines, get_line):
    start_delimiters = ["{", "("]
    end_delimiters = ["}", ")"]
    # Count the start and end delimiters. When the count of both is the same, we are out of the scope.
    count = 0
    while True:
        line = get_line(input_lines)
        for start_delimiter in start_delimiters:
            count += line.count(start_delimiter)
        for end_delimiter in end_delimiters:
            count -= line.count(end_delimiter)
        if count <= 0:
            break 

def comment_out_if_contains_node_type(
    input_lines, output_lines, node_type: str, specializations_to_keep: Iterable[int]
):
    def transform(input_lines):
        input_lines.pop(0)
        return ""

    apply_in_context(
        input_lines,
        output_lines,
        "}",
        lambda line: (
            is_node_type_to_remove(line, node_type)
            and is_spec_to_remove(line, node_type, specializations_to_keep)
        ),
        transform,
    )


def is_node_type_to_remove(line: str, node_types: Iterable[str]):
    for node_type in node_types:
        if re.match(rf".*{node_type}.*", line):
            return True
    return False


def is_spec_to_remove(line: str, node_types: Iterable[str], specializations_to_keep):
    for node_type in node_types:
        if re.match(rf".*{node_type}[a-zA-Z]+", line):
            return False
        specializations = specializations_to_keep[node_type]
        for specialization in specializations:
            node_type_with_spec = rf"{node_type + str(specialization)}"
            if re.match(rf".*{node_type_with_spec}[^0-9]*\s*[(=a-zA-Z,]+", line):
                return False
            node_type_with_spec = rf"{node_type}Node<{specialization}>"
            if re.match(rf".*{node_type_with_spec}[^0-9]*\s*[(=a-zA-Z,]+", line):
                return False
    return True

def parse_node_byte_size(input_lines, output_lines, node_type: str, specializations_to_keep):
    def transform(input_lines):
        traverse_line_scoped(input_lines, lambda input_lines: input_lines.pop(0))
        return ""

    apply_in_context(
        input_lines,
        output_lines,
        "}",
        lambda line: (
            is_node_type_to_remove(line, node_type)
            and is_spec_to_remove(line, node_type, specializations_to_keep)
        ),
        transform,
    )

def parse_make_smallest_leaf_node_for(input_lines, output_lines, node_type: str, specializations_to_keep):
    def transform(input_lines):
        traverse_line_scoped(input_lines, lambda input_lines: input_lines.pop(0))
        return ""

    apply_in_context(
        input_lines,
        output_lines,
        "}",
        lambda line: (
            is_node_type_to_remove(line, node_type)
            and is_spec_to_remove(line, node_type, specializations_to_keep)
        ),
        transform,
    )

def write_smallest_node_type_for_function(node_type: str, input_lines, output_lines, node_types, specializations_to_keep):
    if node_type not in node_types:
        return
    traverse_line_scoped(input_lines, lambda input_lines: input_lines.pop(0)) # Delete the whole function
    node_specializations = specializations_to_keep[node_type]  
    node_specializations = [spec for spec in node_specializations if spec != "Delta"]
    node_specializations.sort()
    output_lines.append(f"/// Returns the smallest leaf node type capable of storing `n` values.\npub fn smallest_{node_type.lower()}_type_for(n: usize) -> VerkleNodeKind {{\nmatch n {{\n")
    last_value = 0
    for i in range(len(node_specializations)):
        specialization = node_specializations[i]
        if i < len(node_specializations) - 1:
            output_lines.append(f"{last_value+1}..={specialization} => VerkleNodeKind::{node_type}{specialization},\n")
            last_value = specialization
        else:
            output_lines.append(f"{last_value+1}..=256 => VerkleNodeKind::{node_type}{specialization},\n")
    output_lines.append("_ => panic!(\"No leaf node type can store more than 256 values\"),\n}\n}\n")   


def parse_mod_rs(node_types, specializations_to_keep):
    MOD_RS_FILE = (
        "/home/luigi-ph3/carmen/rust/src/database/verkle/variants/managed/nodes/mod.rs"
    )
    print(f"Parsing {MOD_RS_FILE} for node type {node_types} and specializations {specializations_to_keep}")
    input_lines = []
    with open(MOD_RS_FILE, "r") as f:
        input_lines = f.readlines()

    output_lines = []
    while len(input_lines) > 0:
        line = input_lines[0].strip()
        if (
            line.startswith("pub enum VerkleNode {")
            or line.startswith(
                "pub fn get_commitment_input(&self) -> BTResult<VerkleCommitmentInput, Error> {"
            )
            or line.startswith("fn to_node_kind(&self) -> Option<Self::Target> {")
            or line.startswith(
                "fn lookup(&self, key: &Key, depth: u8) -> BTResult<LookupResult<Self::Id>, Error> {"
            )
            or line.startswith("pub fn as_inner_node(&self) -> Option<&dyn VerkleManagedInnerNode> {")
            or line.startswith("pub fn accept(")
            or line.startswith("fn next_store_action<'a>(")
            or line.startswith("fn replace_child(&mut self, key: &Key, depth: u8, new: VerkleNodeId) -> BTResult<(), Error> {")
            or line.startswith("fn store(&mut self, update: &KeyedUpdate) -> BTResult<Value, Error> {")
            or line.startswith("fn get_commitment(&self) -> Self::Commitment {")
            or line.startswith("fn set_commitment(&mut self, cache: Self::Commitment) -> BTResult<(), Error> {")
            or line.startswith("pub enum VerkleNodeKind {")
        ):
            print(f"Processing {line}")
            output_lines.append(input_lines.pop(0))
            comment_out_if_contains_node_type(
                input_lines, output_lines, node_types, specializations_to_keep
            )
        elif line.startswith("fn node_byte_size(&self) -> usize {"):
            print(f"Processing {line}")
            output_lines.append(input_lines.pop(0))
            parse_node_byte_size(input_lines, output_lines, node_types, specializations_to_keep)
        elif line.startswith("pub fn make_smallest_leaf_node_for(") or line.startswith("pub fn make_smallest_inner_node_for("):
            print(f"Processing {line}")
            output_lines.append(input_lines.pop(0))
            parse_make_smallest_leaf_node_for(input_lines, output_lines, node_types, specializations_to_keep)
        elif line.startswith("pub fn smallest_leaf_type_for(n: usize) -> VerkleNodeKind {"):
            print(f"Processing {line}")
            write_smallest_node_type_for_function("Leaf",input_lines, output_lines, node_types, specializations_to_keep)
        elif line.startswith("pub fn smallest_inner_type_for(n: usize) -> VerkleNodeKind {"):
            print(f"Processing {line}")
            write_smallest_node_type_for_function("Inner", input_lines, output_lines, node_types, specializations_to_keep)
        else:
            output_lines.append(input_lines.pop(0))

    # Save the new file locally
    with open(MOD_RS_FILE, "w+") as f:
        f.writelines(output_lines)

def parse_lib_rs(node_types, specializations_to_keep):
    LIB_RS_FILE = "/home/luigi-ph3/carmen/rust/src/lib.rs"
    print(f"Parsing {LIB_RS_FILE} for node types {node_types} and specializations {specializations_to_keep}")
    input_lines = []
    with open(LIB_RS_FILE, "r") as f:
        input_lines = f.readlines()
    output_lines = []

    while len(input_lines) > 0:
        line = input_lines[0].strip()
        if line.startswith("pub type VerkleStorageManager = VerkleNodeFileStorageManager<"):
            output_lines.append(input_lines.pop(0))
            def transform(input_lines):
                input_lines.pop(0)
                return ""

            def match(line):
                for node_type in node_types:
                    for specialization in specializations_to_keep[node_type]:
                        node_type_with_spec = rf"{node_type}Node<{specialization}>"
                        if re.match(rf".*{node_type_with_spec}[^0-9]*\s*[(=a-zA-Z,]+", line) or re.match(".*Delta.*", line) or re.match(".*Full.*", line):
                            return False
                return True

            apply_in_context(
                input_lines,
                output_lines,
                ">;",
                lambda line: (
                    match(line)
                ),
                transform,
            )
        else:
            output_lines.append(input_lines.pop(0))

    # Save the new file locally
    with open(LIB_RS_FILE, "w+") as f:
        f.writelines(output_lines)

def parse_id_rs(node_types, specializations_to_keep):
    ID_RS_FILE = "/home/luigi-ph3/carmen/rust/src/database/verkle/variants/managed/nodes/id.rs"
    input_lines = []
    with open(ID_RS_FILE, "r") as f:
        input_lines = f.readlines()
    output_lines = []

    while len(input_lines) > 0:
        line = input_lines[0].strip()
        if line.startswith("fn to_node_kind(&self) -> Option<VerkleNodeKind> {"):
            output_lines.append(input_lines.pop(0))
            comment_out_if_contains_node_type(
                input_lines, output_lines, node_types, specializations_to_keep
            ) 
        if line.startswith("fn from_idx_and_node_kind(idx: u64, node_type: VerkleNodeKind) -> Self {"):
            output_lines.append(input_lines.pop(0))
            comment_out_if_contains_node_type(
                input_lines, output_lines, node_types, specializations_to_keep
            )
        else:
            output_lines.append(input_lines.pop(0))

    # Save the new file locally
    with open(ID_RS_FILE, "w+") as f:
        f.writelines(output_lines)
    

if __name__ == "__main__":
    specializations_names = ["Leaf", "Inner"]
    leaf = [256, "Delta"]
    specializations_to_keep = {
            "Leaf": leaf,
            "Inner": [9, 15, 21, 256, "Delta"],
    }
    parse_mod_rs(
        specializations_names, specializations_to_keep
    )
    parse_lib_rs(
        specializations_names, specializations_to_keep
    )
    parse_id_rs(
        specializations_names, specializations_to_keep
    )
