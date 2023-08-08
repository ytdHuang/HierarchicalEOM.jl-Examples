##############################################################
# Import Packages                                            #
##############################################################
using HierarchicalEOM
using LinearAlgebra
using QuantumOptics # optional (can also construct the operators with standard matrix)

HierarchicalEOM.versioninfo()

##############################################################
# Parameters                                                 #
##############################################################
ϵ  = -5  # energy of electron
U  = 10  # repulsion energy
Γ  = 1   # coupling strength
Wα = 10  # band-width
Φ  = 4   # bias voltage
T  = 0.5 # temperature
Nα = 7   # number of exponent
n_max = 4    # truncation of the fermionic hierarchy
Ith   = 1e-7 # importance threshold

μL =   Φ / 2 # chemical potential of  left-hand side fermionic reservoir
μR = - Φ / 2 # chemical potential of right-hand side fermionic reservoir

##############################################################
# Hamiltonian and Coupling operators                         #
##############################################################
b_spin = SpinBasis(1//2)
σz = sigmaz(b_spin)
σm = sigmap(b_spin)
I2 = identityoperator(b_spin)

# spin-up (-down) annihilation operators
d_up =  σm ⊗ I2
d_dn = -σz ⊗ σm

Hs = ϵ * (d_up' * d_up + d_dn' * d_dn) + U * d_up' * d_up * d_dn' * d_dn


##############################################################
# Construct Bath objects                                     #
##############################################################

# u and d represents spin-up and spin-down, respectively
# L and R represents the left- and right-hand side fermionic reservoir, respectively
fuL = Fermion_Lorentz_Pade(d_up.data, Γ, μL, Wα, T, Nα - 1)
fdL = Fermion_Lorentz_Pade(d_dn.data, Γ, μL, Wα, T, Nα - 1)
fuR = Fermion_Lorentz_Pade(d_up.data, Γ, μR, Wα, T, Nα - 1)
fdR = Fermion_Lorentz_Pade(d_dn.data, Γ, μR, Wα, T, Nα - 1)

# collect all the fermionic bath objects into a list
Fbath = [fuL, fdL, fuR, fdR];

##############################################################
# Construct HEOMLS matrix                                    #
##############################################################

# construct the even-parity HEOMLS (for solving stationary states of ADOs)
L_even = M_Fermion(Hs.data, n_max, Fbath; threshold=Ith)

# construct the odd-parity HEOMLS (for calculating spectrum (density of states) of fermionic system)
L_odd  = M_Fermion(Hs.data, n_max, Fbath, :odd; threshold=Ith)

##############################################################
# Solving stationary states for all ADOs                     #
##############################################################
ados = SteadyState(L_even)

##############################################################
# Calculate density of states under stationary states        #
##############################################################
ωlist = -20:0.4:20
Aω = spectrum(L_odd, ados, d_up.data, ωlist)

##############################################################
# Calculate electronic current with 1st-level-fermionic ADOs #
##############################################################

# a function to calculate electronic current for a given ADOs
function Current(ados, M::M_Fermion)
    
    # the hierarchy dictionary
    HDict = M.hierarchy

    # we need all the indices of ADOs for the first level: [1]
    idx_list = HDict.lvl2idx[1]
    
    Ic = 0.0im # electronic current
    for idx in idx_list
        ρ1 = ados[idx]  # one of the 1st-level ADO

        # find all the corresponding bath index (α) and exponent term index (k)
        nvec = HDict.idx2nvec[idx]
        for (α, k, _) in getIndexEnsemble(nvec, HDict.bathPtr)
            
            # for left-hand side fermionic reservoir
            # α == 1 (spin-up), α == 2 (spin_down)
            if (α == 1) || (α == 2)
                exponent = M.bath[α][k]
                if exponent.types == "fA"     # fermion-absorption
                    Ic += tr(exponent.op' * ρ1)
                elseif exponent.types == "fE" # fermion-emission
                    Ic -= tr(exponent.op' * ρ1)
                end
                break
            end
        end
    end
    
    # change unit to mA
    e = 1.60218e-19
    ħ = 6.62607015e−34 / (2 * π)
    eV_to_Joule = 1.60218e-19  # unit conversion
    
    # (e / ħ) * I  [unit: mA] 
    return (e / ħ) * real(1im * Ic) * eV_to_Joule
end

Current(ados, L_even)
