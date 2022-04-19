import os
import sys

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from cycler import cycler

from helpers.constants import bohr_magneton
from helpers import get_j_str
from atoms import Atom

def get_jraise(j): #defines the raising operator
    dim = round(2*j+1)
    jraise = np.zeros((dim,dim))
    ms = np.linspace(j,-j,dim)
    for i in range(1,dim):
        m = ms[i]
        jraise[i-1,i]=np.sqrt(j*(j+1)-m*(m+1))
    # print(jp)
    return jraise

def get_jvec(j): #defines the general angular momentum matrix for a spin-j system
    dim = round(2*j+1)
    jraise= get_jraise(j)
    jlower = np.transpose(jraise)
    jx = 0.5*(jraise+jlower)
    jy = 1j*0.5*(-jraise+jlower)
    jz = 0.5*(jraise@jlower-jlower@jraise)
    identity = np.zeros((dim,dim))
    for i in range(dim):
        identity[i,i]=1
    return [jx,jy,jz,identity]

def get_J2(l,s): #finds the angular momentum matricies for a two-particle system (or alternatively finds the total J from the L,S vectors)
    L = get_jvec(l)
    S = get_jvec(s)
    Jx = np.kron(L[0],S[3])+np.kron(L[3],S[0])
    Jy = np.kron(L[1],S[3])+np.kron(L[3],S[1])
    Jz = np.kron(L[2],S[3])+np.kron(L[3],S[2])
    J0 = np.kron(L[3],S[3])
    return Jx@Jx + Jy@Jy + Jz@Jz

def get_F2(l,s,i): #finds the F matricies
    L = get_jvec(l)
    S = get_jvec(s)
    Jx = np.kron(L[0],S[3])+np.kron(L[3],S[0])
    Jy = np.kron(L[1],S[3])+np.kron(L[3],S[1])
    Jz = np.kron(L[2],S[3])+np.kron(L[3],S[2])
    J0 = np.kron(L[3],S[3])
    J = [Jx,Jy,Jz,J0]
    I = get_jvec(i)
    Fx = np.kron(J[0],I[3])+np.kron(J[3],I[0])
    Fy = np.kron(J[1],I[3])+np.kron(J[3],I[1])
    Fz = np.kron(J[2],I[3])+np.kron(J[3],I[2])
    F0 = np.kron(J[3],I[3])
    return Fx@Fx + Fy@Fy + Fz@Fz

def H_atom(i,j,A_hfs,B_hfs):
    J = get_jvec(j)
    I = get_jvec(i)

    #0 will be the value of the energy at the fine structure level
    #kron to increase its dimensionality to the |m_i,m_j> basis
    
    IdotJ = (np.kron(I[0],J[3])@np.kron(I[3],J[0])+
             np.kron(I[1],J[3])@np.kron(I[3],J[1])+
             np.kron(I[2],J[3])@np.kron(I[3],J[2]))
    
    identity = np.kron(I[3],J[3])
    
    return (A_hfs*IdotJ+B_hfs*(3*(IdotJ@IdotJ)+3/2*IdotJ-i*(i+1)*j*(j+1)*identity)/(2*i*(2*i-1)*2*j*(2*j-1)))
    

def Lz(l,s,i):
    L = get_jvec(l)
    S = get_jvec(s)
    I = get_jvec(i)
    return np.kron(np.kron(L[2],S[3]),I[3])

def Sz(l,s,i):
    L = get_jvec(l)
    S = get_jvec(s)
    I = get_jvec(i)
    return np.kron(np.kron(L[3],S[2]),I[3])

def Jz(i,j):
    J = get_jvec(j)
    I = get_jvec(i)
    return np.kron(I[3],J[2])

#%%
def eigens(B,atom): #we want to construct the complete Hamiltonian including a magnetic field pertubation
    i = atom.i
    j = atom.j
    A_hfs = atom.A_hfs
    B_hfs = atom.B_hfs
    g_j = atom.g_j   
    
    num_states = round((2*j+1)*(2*i+1))
    fs = np.arange(np.abs(i-j),i+j+1)

    H = H_atom(i,j,A_hfs,B_hfs) + B*(g_j*Jz(i,j))*bohr_magneton
    evals, evecs = np.linalg.eig(H)
    evecs = evecs.transpose() # linalg returns eigenvectors in columns. Switch these to rows.
    evals = np.real(evals) # eigenvalues will be real anyway due to Hermitian Hamiltonian
    return evals,evecs

def sort_eigenvectors(prev_vecs, current_vecs, current_vals):
    """
    As linalg.eig returns eigenvectors in a random order, it is useful to sort them.
    Sorting is implemented by keeping the eigenvalues with the maximal overlap to each
    other together.
    
    Make sure this sorting is done before any further calculations that use the states."""
    sorted_vecs = []
    sorted_vals = []
    for prev_vec in prev_vecs:
        max_prev_overlap = 0
        matching_index = 0
        for veci,current_vec in enumerate(current_vecs):
            prev_overlap = np.abs(np.dot(prev_vec,current_vec))**2/np.abs(np.dot(current_vec,current_vec))**2
            if prev_overlap > max_prev_overlap:
                max_prev_overlap = prev_overlap
                matching_index = veci
        sorted_vecs.append(current_vecs[matching_index])
        sorted_vals.append(current_vals[matching_index])
    return sorted_vecs, sorted_vals

def get_unsplit_energies(A_hfs,B_hfs):
    """Get unsplit energies. Keep only 1 of each value (discard other values in the range A_hfs/10)."""
    unsplit_evals,_ = eigens(0)
    unsplit_evals = unsplit_evals[~(np.triu(np.abs(unsplit_evals[:,None] - unsplit_evals) <= A_hfs/10,1)).any(0)]
    return unsplit_evals

def calc_breit_rabi(Bs,atom,overlap_state=None,num_steps=1000):
    evalss = []
    evecss = []
    overlapss = []
    
    for Bi, B in enumerate(Bs):
        evals, evecs = eigens(B,atom)
                
        if Bi > 0:
            prev_evecs = evecss[-1]
            evecs,evals = sort_eigenvectors(prev_evecs,evecs,evals)
        else:
            idx = evals.argsort()   
            evals = evals[idx]
            evecs = evecs[idx,:]
        
        overlaps = []
        for evec in evecs:
            if overlap_state is not None:
                overlaps.append(np.abs(np.dot(overlap_state,evec))**2/np.abs(np.dot(evec,evec))**2)
            else:
                overlaps.append(0)
                
        evalss.append(list(evals))
        evecss.append(evecs)
        overlapss.append(overlaps)

    evalss = [list(a) for a in list(zip(*evalss))]
    overlapss = [list(a) for a in list(zip(*overlapss))]
    
    return evalss,evecss,overlapss

#%%
def get_basis_index_from_name(m_i,m_j,atom):
    """Returns the index corresponding to the specified |m_i,m_j> state in the basis"""
    i = atom.i
    j = atom.j
    
    basis_len = round((2*j+1)*(2*i+1))
    
    i_num = (basis_len//(2*j+1))*(i-m_i)
    j_num = j-m_j
    
    index = round(i_num+j_num)
    return index

def get_basis_name_from_index(idx,atom):
    i = atom.i
    j = atom.j
    
    basis_len = round((2*j+1)*(2*i+1))
    
    i_num = round(idx//(2*j+1))
    j_num = round(idx%(2*i+1))
    
    mi_val = np.arange(i,-i-1,-1)[i_num]
    mj_val = np.arange(j,-j-1,-1)[j_num]
    
    return r'$|{},{}\rightangle$'.format(get_j_str(mi_val),get_j_str(mj_val))
    
def get_state_index(f,m_f,atom):
    """Return the index of the state of a given |f,m_f>. Note this assumes that the states
    have been sorted by energy at low field (where the |f,m_f> basis is valid)."""
    i = atom.i
    j = atom.j
    g_j = atom.g_j
    
    fs = np.arange(np.abs(i-j),i+j+1)
    
    g_f = g_j*(f*(f+1)-i*(i+1)+j*(j+1))/(2*f*(f+1))
    
    # ASSUMING THAT INCREASING f INCREASES ENERGY (i.e A_hfs > 0).
    
    fs_multiple = []
    for f_temp in fs:
        for mf_temp in np.arange(-f_temp,f_temp+1):
            fs_multiple.append(f_temp)
    index_f = fs_multiple.index(f)
    if g_f > 0:
        index_mf = f+m_f
    else:
        index_mf = f-m_f
    
    index = round(index_f+index_mf)
    return index


def plot_breit_rabi(Bs,atom,state_highlight=None,finish_plot=True,y_scale=18):
    i = atom.i
    j = atom.j
    basis_len = round((2*j+1)*(2*i+1))
    
    highlight_state = np.zeros(basis_len)
    
    if state_highlight != None:
        highlight_index = get_basis_index_from_name(state_highlight[0],state_highlight[1],atom)
        highlight_state[highlight_index] = 1  
    
    evals, evecs, overlaps = calc_breit_rabi(Bs,atom,overlap_state=highlight_state)
    
    # plt.figure(dpi=300)
    
    
    fig, ax = plt.subplots(1, 1)
    fig.set_dpi(300)
    
    for i, (E,overlap) in enumerate(zip(evals,overlaps)):
        if state_highlight != None:
            sc = ax.scatter(Bs,E,c=overlap,s=1,alpha=0.5,vmax=1,vmin=0)#-offset)
            
            # overlap = np.asarray(overlap)
           
            # points = np.array([Bs, E]).T.reshape(-1, 1, 2)
            # segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # # Create a continuous norm to map from data points to colors
            # norm = plt.Normalize(0,1)
            # lc = LineCollection(segments, cmap='viridis', norm=norm)
            # # Set the values used for colormapping
            # lc.set_array(overlap)
            # lc.set_linewidth(4)
            # line = ax.add_collection(lc)
            
        else:
            plt.plot(Bs,E)
    # if state_highlight != None:
    #     cbar = plt.colorbar(sc)
    #     cbar.set_label(r'$|\leftangle ab|\Phi\rightangle|^2$')
    # fig.colorbar(line, ax=ax)
    if state_highlight != None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r'$|\leftangle m_i = {}, m_j = {}|\psi\rightangle|^2$'.format(
            get_j_str(state_highlight[0]),get_j_str(state_highlight[1])))
    plt.ylabel('energy (MHz)')
    plt.xlabel('magnetic field (G)')
    plt.ylim(-atom.A_hfs*y_scale,atom.A_hfs*y_scale)
    plt.xlim(min(Bs),max(Bs))
    plt.title(atom.get_state_label())
    if finish_plot:
        plt.show()
    
def plot_state_decomp(B_decomp,decomp_state,Bs,atom):
    """Decompose the specified state at a particular field into |m_i,m_j> basis.
    The calculation fields Bs should begin at a small enough value that they clearly
    resolve the |f,m_f> regime."""
    
    Bs = np.asarray(Bs)
    B_idx = (np.abs(Bs-B_decomp)).argmin()
    B_decomp = Bs[B_idx]
    # print(B_decomp)
    
    f = decomp_state[0]
    m_f = decomp_state[1]
    
    state_idx = get_state_index(f,m_f,atom)
    # print(state_idx)
    
    evalss,evecss,_ = calc_breit_rabi(Bs,atom)
    energy = evalss[state_idx][B_idx]
    state = evecss[B_idx][state_idx]
    
    i = atom.i
    j = atom.j
    basis_len = round((2*j+1)*(2*i+1))
    
    # print(state)
    decomp = []
    for overlap in state:
        decomp.append(np.abs(overlap)**2)
    barlist = plt.bar(np.arange(basis_len),decomp)#,label=j,alpha=0.7)
    for i,bar in enumerate(barlist):
        # print(i)
        bar.set_color('C{}'.format(i))
    
    labels = []
    for basis_idx in np.arange(basis_len):
        labels.append(get_basis_name_from_index(basis_idx,atom))
        
    # print(labels)
    plt.xticks(np.arange(basis_len),labels=labels,rotation='vertical')
    plt.title(r'state decomposition of $|\psi\rightangle\sim|f,m_f\rangle = |{},{}>$ at {:.2f} G'.format(f,m_f,B_decomp))
    plt.ylabel('probability amplitude')
    plt.xlabel(r'$|\leftangle\psi|m_i,m_j\rangle|^2$')
    # plt.xlim(left=7.5,right=23.5)
    plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.show()

    
#%%
if __name__ == '__main__':
    atom = Atom('Rb','E2')
    Bs = np.linspace(0.5,500,1000) # don't start at exactly zero otherwise state labelling gets confused
    
    plot_breit_rabi(Bs,atom,state_highlight=(0.5,0.5),y_scale=18)
    
    plot_breit_rabi(Bs,atom,y_scale=18)
    
    plot_state_decomp(100,(1,1),Bs,atom)
    plot_state_decomp(100,(2,1),Bs,atom)
