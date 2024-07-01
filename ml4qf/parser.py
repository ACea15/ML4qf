def parse_textfile(file_path: str) -> dict:
    d1 = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or '=' not in line:
                continue
            # Split the line by the equals sign
            left_side, right_side = line.split('=', 1)
            right_side = right_side.strip()
            
            # Split the left side by the underscore
            container_var = left_side.strip().split('_')
            if len(container_var) == 2:
                container, variable = container_var
                # Update the dictionary
                if container not in d1:
                    d1[container] = {}
                d1[container][variable] = right_side
            # variable with no container    
            elif len(container_var) == 1:
                variable = container_var
                d1[variable[0]] = right_side
            else:
                raise ValueError(f"Incorrect input: {container_var}")
    
    return d1


# # Example usage
# file_path = '/home/acea/projects/XQFM/examples/prototyping/joshicpp/input4.txt'
# result_dict = parse_textfile(file_path)
# print(result_dict)
