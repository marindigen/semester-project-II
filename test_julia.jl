using HDF5

# Specify the path to your HDF5 file
pathdata = "./data/ising_chains.h5"
output_path = "./output.txt"

# Open the HDF5 file and list all datasets
open(output_path, "w") do output_file
    h5open(pathdata, "r") do file
        println("Datasets in the HDF5 file:")
        function list_datasets(parent, indent="")
            for name in keys(parent)
                if typeof(parent[name]) <: HDF5.Group
                    println("$indent Group: $name")
                    list_datasets(parent[name], indent * "  ")
                else
                    println("$indent Dataset: $name")
                end
            end
        end
        list_datasets(file)
    end
end

