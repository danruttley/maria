if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import get_state_label

atom_properties = {'Rb': {'A':87, # mass number
                          'Z':37, # atomic number
                          'i':3/2, # nuclear spin
                          's':1/2, # electron spin
                          'g_s':2.0023193043737, # electron spin g-factor
                          'g_l':0.99999369, # electron orbital g-factor
                          'g_i':-0.0009951414, # nuclear g-factor
                          'hyperfine':{'ground':{'A_hfs':3417.34130545215, # magnetic dipole constant (MHz)
                                                 'B_hfs':0, # electric quadrupole constant (MHz)
                                                 'n':5, # principle quantum number
                                                 'l':0, # electron oribital angular momentum
                                                 'j':0.5 # electron angular momentum
                                                 },
                                       'D1':{'A_hfs':408.328, # magnetic dipole constant (MHz)
                                             'B_hfs':0, # electric quadrupole constant (MHz)
                                             'n':5, # principle quantum number
                                             'l':1, # electron oribital angular momentum
                                             'j':0.5 # electron angular momentum
                                             },
                                       'D2':{'A_hfs':84.7185, # magnetic dipole constant (MHz)
                                             'B_hfs':12.4965, # electric quadrupole constant (MHz)
                                             'n':5, # principle quantum number
                                             'l':1, # electron oribital angular momentum
                                             'j':1.5 # electron angular momentum
                                             },
                                       'E1':{'A_hfs':132.552, # magnetic dipole constant (MHz)
                                             'B_hfs':0, # electric quadrupole constant (MHz)
                                             'n':6, # principle quantum number
                                             'l':1, # electron oribital angular momentum
                                             'j':0.5 # electron angular momentum
                                             },
                                       'E2':{'A_hfs':27.700, # magnetic dipole constant (MHz)
                                             'B_hfs':3.953, # electric quadrupole constant (MHz)
                                             'n':6, # principle quantum number
                                             'l':1, # electron oribital angular momentum
                                             'j':1.5 # electron angular momentum
                                             }
                                       }
                          }
                   }

class Atom():
    
    def __init__(self,species='Rb',transition='E2'):
        """
        Container function for atoms which allows for easy retrieval of
        their properties.

        Parameters
        ----------
        species : string, optional
            The species of atom to use. Either 'Rb' or 'Cs'.
            The default is 'Rb'.
        transition : string, optional
            The transition to use. Labelled analogous to the D1 and D2 lines.
            A higher letter indicates excitation to a higher nP state.
            The default is 'E2' (5S -> 6P_{3/2} transition in Rb).

        Returns
        -------
        None.

        """
        
        self.species = species
        self.transition = transition
        self.set_attributes()
        
    def set_attributes(self):
        properties = atom_properties[self.species]
        for key in properties:
            if key == 'hyperfine':
                hyperfine_properties = properties['hyperfine'][self.transition]
                for hyperfine_key in hyperfine_properties:
                    setattr(self,hyperfine_key,hyperfine_properties[hyperfine_key])
            else:
                setattr(self,key,properties[key])
        self.g_j = (self.g_l*(self.j*(self.j+1)-self.s*(self.s+1)+self.l*(self.l+1))/(2*self.j*(self.j+1))+
                    self.g_s*(self.j*(self.j+1)+self.s*(self.s+1)-self.l*(self.l+1))/(2*self.j*(self.j+1)))
        
    def get_state_label(self):
        return get_state_label(self.species,self.n,self.l,self.j)
        