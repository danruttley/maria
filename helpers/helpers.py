l_names = {0:'s',1:'p',2:'d',3:'f',4:'g'}

def get_j_str(j):
    if j%1 == 0:
        return str(j)
    else:
        return '{}/2'.format(int(j*2))

def get_state_label(species,n,l,j,mj=None):
    if mj == None:
        state_label = '{} |{}'.format(species,n)+l_names[l]+r'$_{%s}\rightangle$'%(get_j_str(j))
    else:
        state_label = '{} |{}'.format(species,n)+l_names[l]+r'$_{%s}$, $m_j = %s$'%(get_j_str(j),get_j_str(mj))+r'$\rightangle$'
    return state_label