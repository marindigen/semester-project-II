using PlmDCA
using HDF5
using JLD

pathdata = "./data/ising_chains.h5"
fid = h5open(pathdata,"r")
finalchains = read(fid,"train")
finalchains = (finalchains .+ 1)/2
finalchains = finalchains .+ 1
finalchains = Int.(finalchains)
weights = ones(nbrseq)/nbrseq
temperatures = read(fid,"temperatures")
nbrspin, nbrseq, nbrtemperatures = size(finalchains)
Jtensors = Array{Float64}(undef,nbrtemperatures,2,2,nbrspin,nbrspin)
weights = ones(nbrseq)/nbrseq

for i =  1:length(temperatures)
     X_default = plmdca_asym(finalchains[:,:,i], weights,lambdaJ = 0.01,lambdaH = 0.01, verbose = false)
     Jtensors[i,:,:,:,:] = X_default.Jtensor
end
pathsavefolder = "./results/"
save(pathsavefolder*"plmDCA_couplings.jld","couplings",Jtensors)

