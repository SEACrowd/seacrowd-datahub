import re


def extract_parts(input_string):
    parts = []
    stack = []
    current_part = ""

    for char in input_string:
        if char == "(":
            stack.append("(")
        elif char == ")":
            if stack:
                stack.pop()
                if not stack:
                    parts.append(current_part[1:].strip())
                    current_part = ""
            else:
                parts.append(current_part[1:].strip())
                current_part = ""
        if stack:
            current_part += char

    return parts


def extract_sentence(input_string):
    innermost_pattern = re.compile(r"\(([^()]+)\)")
    innermost_matches = re.findall(innermost_pattern, input_string)
    extracted_sentence = " ".join(match.split()[1] for match in innermost_matches)
    if len(extracted_sentence) == 0:
        extracted_sentence = " ".join(input_string.split()[1:])
    return extracted_sentence


def extract_data(sentence):
    nodes = []
    sub_nodes = {}
    sub_node_ids = []

    # Extract id, sub_nodes and text of ROOT
    sentence_id = sentence.split("\t")[0]
    root_sent = sentence[sentence.find("ROOT") : -1]
    root_subnodes = extract_parts(root_sent)
    sub_nodes.update({i + 1: root_subnodes[i] for i in range(len(root_subnodes))})
    sub_node_ids.extend([i + 1 for i in range(len(root_subnodes))])
    root_text = extract_sentence(root_sent)

    nodes.append({"id": f"{sentence_id+'.'+str(0)}", "type": "ROOT", "text": root_text, "offsets": [0, len(root_text) - 1], "subnodes": [f"{sentence_id+'.'+str(i)}" for i in sub_node_ids]})

    while sub_node_ids:
        sub_node_id = sub_node_ids.pop(0)
        text = extract_sentence(sub_nodes[sub_node_id])

        cur_subnodes = extract_parts(sub_nodes[sub_node_id])

        if len(cur_subnodes) > 0:
            id_to_add = sub_node_ids[-1] if len(sub_node_ids) > 0 else sub_node_id
            cur_subnode_ids = [id_to_add + i + 1 for i in range(len(cur_subnodes))]
            sub_nodes.update({id_to_add + i + 1: cur_subnodes[i] for i in range(len(cur_subnodes))})
            sub_node_ids.extend(cur_subnode_ids)
        else:
            cur_subnode_ids = []

        node_type = sub_nodes[sub_node_id].split(" ")[0]
        start = root_text.find(text)
        end = start + len(text) - 1

        nodes.append({"id": f"{sentence_id+'.'+str(sub_node_id)}", "type": node_type, "text": text, "offsets": [start, end], "subnodes": [f"{sentence_id+'.'+str(i)}" for i in cur_subnode_ids]})
    return {"id": sentence_id, "passage": {"id": sentence_id + "_0", "type": None, "text": [nodes[0]["text"]], "offsets": nodes[0]["offsets"]}, "nodes": nodes}
