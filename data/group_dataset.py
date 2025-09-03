import numpy as np 

class DMU_group():
    """
    This is one group of n_dmus, n_inputs, n_outputs 
    """
    
    def __init__(self,n_dmus,n_inputs,n_outputs,input_range,output_range):
        self.n_dmus = n_dmus
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.input_range = input_range
        self.output_range = output_range
        
        self.efficinces = []
        
        self.inputs = np.zeros(n_inputs)
        self.outputs = np.zeros(n_outputs)
        
        
    def __getitem__():
        #TODO: NA FTIAKSW TIN GET ITEM ETSI WSTE NA EINAI BATCH SIZE KAI COMPATIBLE ME TO MODEL
        #TO MODEL THA EINAI SEQUENCE TO SEQUENCE OPOTE EXE LIGO TO NOU SOU 
        return 0
    
    
    def data_generator(self):
        """
        
        Parameters
        ----------
        
        n_dmus : int 
            number of DMUs that one group has     
        n_inputs : int
            number of input features of the DMUs
        n_outputs : int
            number of output features of the DMUs
        input_range : list
            contains the range of the values of the inputs ex. [1,10]
        output_range : list
            contains the range of the values of the outputs ex. [5,40]

        Returns
        -------
        2D numpy array of dimensions n_dmus x (n_input + n_outputs) 
        
        example
        n_dmus = 4
        n_inputs = 3
        n_outputs = 2
        
        ---------------- inputs --------------|--------outputs--------
        [[ 4.51686484  7.81003239  3.19986647 |18.65738797 14.31297242]
         [ 6.76773995  3.89073557  8.61425806 |13.81724567 13.29757381]
         [ 6.4705536   6.5266032   1.02544129 | 6.12165456 11.80065996]
         [ 6.12715881  8.32606671  5.1681374  | 11.37134163 17.99563935]]
        
        """
        self.inputs = np.random.uniform(self.input_range[0],self.input_range[1],size = (self.n_dmus,self.n_inputs))
        self.outputs = np.random.uniform(self.output_range[0],self.output_range[1],size = (self.n_dmus,self.n_outputs))
        
    def show_content():
        pass
    
    def calculate_efficiences(self):
        return self.efficinces
        