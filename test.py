from utils import average_pairwise_distance
from parse import read_input_file, read_output_file

if __name__ == "__main__":
    output_dir = "outputs"
    input_dir = "inputs"
    input_path = "medium-101.in"
    graph_name = input_path.split(".")[0]
    G = read_input_file(f"{input_dir}/{input_path}")
    T = read_output_file(f"{output_dir}/{graph_name}.out", G)
    print(average_pairwise_distance(T))
