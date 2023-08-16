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
ϵ  = -3    # energy of electron
ωc =  1    # frequency of single-mode cavity
g  =  0.5  # electron-cavity coupling strength
Γ  =  1    # electron-fermionic reservoir coupling strength 
Δ  =  0.01 #   cavity-bosonic   reservoir coupling strength 
Wβ =  0.2  # band-width of bosonic reservoir
Wα = 10    # band-width of fermionic reservoir
Φ  =  6    # bias voltage
T  =  0.5  # temperature
Np =  6    # truncation of the cavity photon number
Nβ =  5    # number of exponent for bosonic   reservoir
Nα =  7    # number of exponent for fermionic reservoir
m_max = 4    # truncation of the bosonic   hierarchy
n_max = 3    # truncation of the fermionic hierarchy
Ith   = 1e-6 # importance threshold

μL =   Φ / 2 # chemical potential of  left-hand side fermionic reservoir
μR = - Φ / 2 # chemical potential of right-hand side fermionic reservoir

##############################################################
# Hamiltonian and Coupling operators                         #
##############################################################
b_spin = SpinBasis(1//2)
b_phot = FockBasis(Np)

σm = sigmap(b_spin)
Is = identityoperator(b_spin)
Ip = identityoperator(b_phot)

# photon / electron annihilation operators
a = destroy(b_phot) ⊗ Is
d = Ip              ⊗ σm

He = ϵ  * d' * d
Hc = ωc * a' * a
Hs = Hc + He + g * d' * d * (a + a');


##############################################################
# Construct Bath objects                                     #
##############################################################
# L and R represents the left- and right-hand side fermionic reservoir, respectively
fL = Fermion_Lorentz_Pade(d.data, Γ, μL, Wα, T, Nα - 1)
fR = Fermion_Lorentz_Pade(d.data, Γ, μR, Wα, T, Nα - 1)

# collect all the fermionic bath objects into a list
Fbath = [fL, fR];

# boson baths
Bbath = Boson_DrudeLorentz_Pade((a + a').data, Δ, Wβ, T, Nβ - 1);

##############################################################
# Construct HEOMLS matrix                                    #
##############################################################
# construct the even-parity HEOMLS for 
## 1. solving stationary states of ADOs
## 2. calculating spectrum (power spectral density) of bosonic system
L_even = M_Boson_Fermion(Hs.data, m_max, n_max, Bbath, Fbath; threshold=Ith)

# construct the odd-parity HEOMLS for calculating spectrum (density of states) of fermionic system
L_odd  = M_Boson_Fermion(Hs.data, m_max, n_max, Bbath, Fbath, :odd; threshold=Ith)

##############################################################
# Construct HEOMLS matrix (with Master Equation approach)    #
##############################################################
# Drude-Lorentz spectral density
Jβ(ω) = (4 * Δ * Wβ * ω) / (ω ^ 2 + Wβ ^ 2)

# Bose-Einstein distribution
nβ(ω) = (exp(ω / T) - 1) ^ (-1)

# the list of jump operators
Jop = [
    √(Jβ(ωc) * (nβ(ωc) + 1)) * (a).data,
    √(Jβ(ωc) *  nβ(ωc)     ) * (a').data
]

# remove the bosonic hierarchy and add Lindbladian to the HEOMLS
L_ME = M_Fermion(Hs.data, n_max, Fbath)
L_ME = addBosonDissipator(L_ME, Jop)

##############################################################
# Solving stationary states for all ADOs                     #
##############################################################
# with HEOM approach (bosonic env.)
ados_HEOM = SteadyState(L_even) 

# with Lindblad Master equation approach (bosonic env.)
ados_ME   = SteadyState(L_ME)  

##############################################################
# Calculate density of states under stationary states        #
##############################################################
ωlist = -6:0.06:0
Aω = spectrum(L_odd, ados_HEOM, d.data, ωlist)

##############################################################
# Calculate power spectral density under stationary states   #
##############################################################
ωlist = 0:0.2:6

# with HEOM approach (bosonic env.)
Sω_HEOM = spectrum(L_even, ados_HEOM, a.data, ωlist[2:end])

# with Lindblad Master equation approach (bosonic env.)
Sω_ME   = spectrum(L_ME,   ados_ME,   a.data, ωlist[2:end])

##############################################################
# Calculate electronic current with 1st-level-fermionic ADOs #
##############################################################
# a function to calculate electronic current for a given ADOs
function Current(ados, M::M_Boson_Fermion)
    
    # the hierarchy dictionary
    HDict = M.hierarchy

    # we need all the indices of ADOs for the first level: [1]
    idx_list = HDict.Flvl2idx[1]
    
    Ic = 0.0im # electronic current
    for idx in idx_list
        ρ1 = ados[idx]  # one of the 1st-level ADO

        # with bosonic level = 0
        # find all the corresponding fermionic bath index (α) and exponent term index (k)
        nvec_b, nvec_f = HDict.idx2nvec[idx]
        if nvec_b.level == 0
            for (α, k, _) in getIndexEnsemble(nvec_f, HDict.fermionPtr)

                # α == 1 (left-hand side fermionic reservoir)
                if α == 1
                    exponent = M.Fbath[α][k]
                    if exponent.types == "fA"     # fermion-absorption
                        Ic += tr(exponent.op' * ρ1)
                    elseif exponent.types == "fE" # fermion-emission
                        Ic -= tr(exponent.op' * ρ1)
                    end
                    break
                end
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
