
config={"res_ch": 0, # --Fixed
        "ro_chs": [0], # --Fixed
        "reps":1, # --Fixed
        "relax_delay": 1.0, # --us
        "res_phase": 0, # --degrees
        "pulse_style": "const",         
        "length":20, # [Clock ticks]        
        "readout_length":100, # [Clock ticks]
        "pulse_gain":3000, # [DAC units]
        "pulse_freq": 250, # [MHz] 
        "adc_trig_offset": 100, # [Clock ticks] 
        "soft_avgs":100
       }
D


class LoopbackProgram(AveragerProgram):
    def initialize(self):
        cfg=self.cfg   
        res_ch = cfg["res_ch"]
        
        self.declare_gen(ch=cfg["res_ch"], nqz=1)
        
        for ch in cfg["ro_chs"]:
            self.declare_readout(ch=ch, length=self.cfg["readout_length"],
                                    freq=self.cfg["pulse_freq"], gen_ch=cfg["res_ch"])

        freq = self.freq2reg(cfg["pulse_freq"],gen_ch=res_ch, ro_ch=cfg["ro_chs"][0])
        phase = self.deg2reg(cfg["res_phase"], gen_ch=res_ch)
        gain = cfg["pulse_gain"]
        self.default_pulse_registers(ch=res_ch, freq=freq, phase=phase, gain=gain)

        style=self.cfg["pulse_style"]

        if style in ["flat_top","arb"]:
            sigma = cfg["sigma"]
            self.add_gauss(ch=res_ch, name="measure", sigma=sigma, length=sigma*5)
            
        if style == "const":
            self.set_pulse_registers(ch=res_ch, style=style, length=cfg["length"])
        elif style == "flat_top":
            self.set_pulse_registers(ch=res_ch, style=style, waveform="measure", length=cfg["length"])
        elif style == "arb":
            self.set_pulse_registers(ch=res_ch, style=style, waveform="measure")
        
        self.synci(200)  # give processor some time to configure pulses
    
    def body(self):
        self.trigger(adcs=self.ro_chs,
                        pins=[0], 
                        adc_trig_offset=self.cfg["adc_trig_offset"])
        self.pulse(ch=self.cfg["res_ch"])
        self.wait_all()
        self.sync_all(self.us2cycles(self.cfg["relax_delay"]))       

prog =LoopbackProgram(soccfg, config)
iq_list = prog.acquire_decimated(soc, load_pulses=True, progress=True)   
