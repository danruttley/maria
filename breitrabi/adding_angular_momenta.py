import os
import sys

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
import numpy as np
import matplotlib.pyplot as plt

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

def H_atom(l,s,i):
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
    
    J2 = Jx@Jx + Jy@Jy + Jz@Jz
    F2 = Fx@Fx + Fy@Fy + Fz@Fz
    
    #Add fine/hyperfine structure splitting
    A_hfs = 20
    A_fs = 200000*A_hfs #NB: In reality this would be about 2000*A_hfs

    #0 will be the value of the energy at the coarse structure level
    #kron J2 with the I[4] (identity) to increase its dimensionality to the |m_l,m_s,m_I> basis
    #l(l+1) is the eigenvalue of the L^2 matrix, same for s(s+1) and S^2 and i(i+1) for I^2
    return A_fs*(np.kron(J2,I[3])-l*(l+1)*F0-s*(s+1)*F0) + A_hfs*(F2-np.kron(J2,I[3])-i*(i+1)*F0)

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


plt.close()

# jvec = get_jvec(3/2)

# for i in range(4):
#     print(jvec[i])
    
# J2 = get_J2(0.5,0.5)
# print(J2)

# F2 = np.real(get_F2(1,0.5,0.5))
# # plt.imshow(F2)
# # plt.show()

# H = H_atom(1,0.5,0.5)
# print(H)
# # plt.imshow(np.real(H))
# # plt.show()

#%% 6P3/2
l = 1
s = 0.5
i = 1.5
num_states = round((2*l+1)*(2*s+1)*(2*i+1))



gL = 1 # electron orbital g-factor
gS = 2 # electron spin g-factor

def get_name_from_index(i):  
    try:
        mj = np.round(np.floor(i/4))-7/2
        mi = i%4-3/2
        if mj < 0:
            mi *= -1
        # mi = mf - mj
        return '|{:.1f}, {:.1f}>'.format(mj,mi)
    except Exception:
        i = list(i)
        print(i)
        names = []
        for j in i:
            name = get_name_from_index(j)
            print(name)
            names.append(name)
        print(names)
        return names

def eigens(B): #we want to construct the complete Hamiltonian including a magnetic field pertubation
    H = H_atom(l,s,i) + B*(Lz(l,s,i)+2*Sz(l,s,i))
    evals, evecs = np.linalg.eig(H)
    return evals,evecs

Bs = np.linspace(0,300,200)
evalss = []
evecss = []
# Bs = [0,1e-6,1e5] #uncomment for decompositon calcs
for B in Bs:
    evals, evecs = eigens(B)
    idx = evals.argsort()   
    evals = evals[idx]
    evecs = evecs[:,idx]
    
    evalss.append(list(evals))
    evecss.append(evecs)

offset_list = [np.real(x) for x in evalss[0]]
offset_list.sort()
offset = offset_list[8]

evalss = [list(a) for a in list(zip(*evalss))]

plt.figure(dpi=300)
for i, E in enumerate(evalss):
    if i > 7:
        plt.plot(Bs,E-offset,c='C{}'.format(i))
        plt.text(Bs[-1],E[-1]-offset,get_name_from_index(i),ha='left',va='center',size='xx-small',c='C{}'.format(i))
        if i in [8,10,14,20]:
            Fs = {8:0,10:1,14:2,20:3}
            plt.text(Bs[0],E[0]-offset,'F = {}'.format(Fs[i]),ha='right',va='center',size='xx-small',c='k'.format(i))
plt.ylabel('energy (arb.)')
plt.xlabel('magnetic field (arb.)')
plt.show()

plt.figure()
B_index = 1
state_index = 11
f = 1
mf = 1
state_name = '|F,mF> = |{},{}>'.format(f,mf)


state = evecss[B_index][:,state_index]
print(state)
decomp = []
for i in range(num_states):
    decomp.append(np.abs(np.dot(state,evecss[-1][:,i]))**2)
barlist = plt.bar(np.arange(num_states),decomp)#,label=j,alpha=0.7)
for i,bar in enumerate(barlist):
    print(i)
    bar.set_color('C{}'.format(i))
plt.xticks(np.arange(num_states),labels=get_name_from_index(np.arange(num_states)),rotation='vertical')
plt.title(r'state decomposition of $|F,m_F\rangle = |{},{}>$'.format(f,mf))
plt.ylabel('probability amplitude')
plt.xlabel(r'$|m_J,m_I\rangle$')
plt.xlim(left=7.5,right=23.5)
plt.tight_layout()
plt.show()

print(state_name)
total_prob = 0
for i, prob in enumerate(decomp):
    total_prob += prob
    if prob > 0.01:
        print('{} \t{:.2f}'.format(get_name_from_index(i),prob))
print('TOTAL         \t{:.2f}'.format(total_prob))

#%% 5S1/2
l = 0
s = 0.5
i = 1.5
num_states = round((2*l+1)*(2*s+1)*(2*i+1))

def get_name_from_index(i):  
    try:
        mj = np.round(np.floor(i/4))-1/2
        mi = i%4-3/2
        if mj < 0:
            mi *= -1
        # mi = mf - mj
        return '|{:.1f}, {:.1f}>'.format(mj,mi)
    except Exception:
        i = list(i)
        print(i)
        names = []
        for j in i:
            name = get_name_from_index(j)
            print(name)
            names.append(name)
        print(names)
        return names

def eigens(B): #we want to construct the complete Hamiltonian including a magnetic field pertubation
    H = H_atom(l,s,i) + B*(Lz(l,s,i)+2*Sz(l,s,i))
    evals, evecs = np.linalg.eig(H)
    return evals,evecs

Bs = np.linspace(0,300,200)
evalss = []
evecss = []
# Bs = [0,1e-6,1e5] #uncomment for decompositon calcs
for B in Bs:
    evals, evecs = eigens(B)
    idx = evals.argsort()   
    evals = evals[idx]
    evecs = evecs[:,idx]
    
    evalss.append(list(evals))
    evecss.append(evecs)

offset_list = [np.real(x) for x in evalss[0]]
offset_list.sort()
offset = offset_list[0]

evalss = [list(a) for a in list(zip(*evalss))]

plt.figure(dpi=100)
for i, E in enumerate(evalss):
    plt.plot(Bs,E-offset,c='C{}'.format(i))
    plt.text(Bs[-1],E[-1]-offset,get_name_from_index(i),ha='left',va='center',size='xx-small',c='C{}'.format(i))
    if i in [0,4]:
        Fs = {0:1,4:2}
        plt.text(Bs[0],E[0]-offset,'F = {}'.format(Fs[i]),ha='right',va='center',size='xx-small',c='k'.format(i))
plt.ylabel('energy (arb.)')
plt.xlabel('magnetic field (arb.)')
plt.show()

plt.figure()
B_index = 1
state_index = 0
f = 1
mf = 1
state_name = '|F,mF> = |{},{}>'.format(f,mf)

state = evecss[B_index][:,state_index]
print(state)
decomp = []
for i in range(num_states):
    decomp.append(np.abs(np.dot(state,evecss[-1][:,i]))**2)
barlist = plt.bar(np.arange(num_states),decomp)#,label=j,alpha=0.7)
for i,bar in enumerate(barlist):
    print(i)
    bar.set_color('C{}'.format(i))
plt.xticks(np.arange(num_states),labels=get_name_from_index(np.arange(num_states)),rotation='vertical')
plt.title(r'state decomposition of $|F,m_F\rangle = |{},{}>$'.format(f,mf))
plt.ylabel('probability amplitude')
plt.xlabel(r'$|m_J,m_I\rangle$')
plt.xlim(left=-0.5,right=7.5)
plt.tight_layout()
plt.show()

print(state_name)
total_prob = 0
for i, prob in enumerate(decomp):
    total_prob += prob
    if prob > 0.01:
        print('{} \t{:.2f}'.format(get_name_from_index(i),prob))
print('TOTAL         \t{:.2f}'.format(total_prob))

