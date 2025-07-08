import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from .SONOS import SONOS


class customSONOS(SONOS):
    # override any methods or add new ones here
    # any methods that aren't overriden are inherited
    # from SONOS class in SONOS.py script

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # <- should always be called
        #prog_params = self.device_params.programming_error
        # if self.device_params.programming_error.model == "CustomSONOS":

        #self.TID_amount = getattr(prog_params, "TID_amount", 0)
        #self.alpha_mu = getattr(prog_params, "alpha_mu", 1.0)
        #self.alpha_sig = getattr(prog_params, "alpha_sig", 1.0)
        #self.std_csv_loc = getattr(prog_params, "std_csv", None)
        #self.shift_csv_loc = getattr(prog_params, "shift_csv", None)
        
        self.TID_amount = self.device_params.programming_error.TID_amount
        self.shift_csv_loc = self.device_params.programming_error.shift_csv_loc
        self.std_csv_loc = self.device_params.programming_error.std_csv_loc
        self.alpha_mu = self.device_params.programming_error.alpha_mu
        self.alpha_sig = self.device_params.programming_error.alpha_sig
        self.tidData = False

        if None not in (self.shift_csv_loc, self.std_csv_loc):
            self.mu_df = pd.read_csv(self.shift_csv_loc)
            self.sig_df = pd.read_csv(self.std_csv_loc)
 
            self.raw_mu = self._build_raw(self.mu_df, "mu")
            self.raw_sig = self._build_raw(self.sig_df, "sig")
            self.interp_mu = self._build_interp(self.raw_mu)
            self.interp_sig = self._build_interp(self.raw_sig)
            self.tidData = True # <- TID response data is present (and correct path)

    def _build_raw(self, df, type):
        raw = {}
        for col in df.columns:
            if not col.endswith("_G0"):
                continue
            dose = int(col.split("_")[0])
            G0_arr = df[col].to_numpy()
            val_arr = df[f"{dose}_{type}"].to_numpy()
            
            mask = ~np.isnan(G0_arr) & ~np.isnan(val_arr)
            raw[dose] = (G0_arr[mask], val_arr[mask])
        return raw
    
    def _build_interp(self, raw_dict):
        # building interpolators using extracted G0 (initial) and I_w shift (mu) values
        # (d = dose level, xi = G0 value) y = I_w shift (mu)
        pts, vals = [], []
        for d, (x, y) in raw_dict.items():
            pts += [(d, xi) for xi in x]
            vals += list(y)
        return LinearNDInterpolator(np.array(pts), np.array(vals))


    def _interpolate_TID(self, I):
        def interp_one(val):

            mu = self.interp_mu(self.TID_amount, val) #* self.alpha_mu
            sig = self.interp_sig(self.TID_amount, val) #* self.alpha_sig
            if np.isnan(mu) or np.isnan(sig):
                mu = 0.0
                sig = 0.0

            delta_I = np.random.normal(loc=mu, scale=sig) # <- finding random value one  
                                                          #    std around mean shift in I_w
            I_prime = np.clip(val + delta_I, self.Imin, self.Imax)
            return I_prime
                
        interp_vec = np.vectorize(interp_one)
        return interp_vec(I)
    
    def programming_error(self, input_):
        result = super().programming_error(input_)
        
        if self.TID_amount > 0 and self.tidData:
            I = self._calculate_current(result)
            I_prime = self._interpolate_TID(I)
            input_ = self.Gmin_norm + self.Grange_norm * (I_prime - self.Imin) / (
                self.Imax - self.Imin
            ) # input_ results in normalized G_val.  
        return input_

    def drift_error(self, input_, time):
        result = super().drift_error(input_, time)
        # ...custom code...
        return result
    
    def read_noise(self, input_):
        result = super().read_noise(input_)
        #...custom code...
        return result