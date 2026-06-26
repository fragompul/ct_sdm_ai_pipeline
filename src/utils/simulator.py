# src/utils/simulator.py

import time
import numpy as np
import math
import logging

try:
    import cbadc as cb
except ImportError:
    logging.warning("Librería cbadc no encontrada en este entorno.")

class CBADCSimulator:
    """Wrapper completo para validar las predicciones con el simulador real."""
    
    def __init__(self, mapping_info=None):
        self.mapping_info = mapping_info

    def simulate(self, topology_id, design_vars_array, target_specs, feature_names, top_name=""):
        """
        Recibe el vector predicho, reconstruye las variables y lanza cbadc.
        """
        start_time = time.time()
        dict_vars = dict(zip(feature_names, design_vars_array))
        
        # 1. Deducir arquitectura desde el nombre (ej. "fb_2_active_rc")
        top_name = top_name.lower()
        form = "FB" if "fb" in top_name else "FF"
        order = 2
        if "_3_" in top_name: order = 3
        elif "_4_" in top_name: order = 4
        
        implementation = "Active_RC" if "active_rc" in top_name else "Gm_C"
        
        try:
            # 2. Extraer parámetros base
            Bw = target_specs['Bw']
            osr = int(round(dict_vars.get("osr", 64)))
            nlev = int(round(dict_vars.get("nlev", 2)))
            Hinf = dict_vars.get("Hinf", 1.5)
            
            fs = Bw * osr * 2
            dt = 1.0 / fs
            
            # 3. Sintetizar NTF y Frontend Analógico Ideal
            ntf = cb.delsig.synthesizeNTF(order, osr, 2, Hinf, 0.0)
            ABCDc, tdac2 = cb.delsig.realizeNTF_ct(ntf, form, [0, 1])
            analog_frontend = cb.AnalogFrontend.ctsdm(ABCDc, tdac2, dt, nlev)
            analog_frontend.dt = dt
            analog_frontend.output_covariance = np.full((analog_frontend.M, analog_frontend.M), 1e-12)
            
            # Función auxiliar para recuperar arrays de tamaño "order"
            def get_array(name, default_val=1e-12):
                arr = []
                for i in range(1, order + 1):
                    val = dict_vars.get(f"{name}{i}")
                    if val is None or math.isnan(val):
                        val = dict_vars.get(name, default_val)
                    arr.append(val)
                return np.array(arr)

            # Extraer Offset
            analog_frontend.V_offset = get_array("V_offset", 0.0)

            # 4. Construir la Implementación Física
            if implementation == "Active_RC":
                Cint = get_array("Cint", 1e-12)
                gm = get_array("gm", 1e-4)
                Ro = get_array("Ro", 1e5)
                Co = get_array("Co", 1e-12)
                slew_rate = get_array("slew_rate", 1e7)
                
                analog_frontend_impl = cb.ActiveRC(
                    analog_frontend, Cint, gm, Ro, Co, slew_rate
                )
            else:
                Cint = get_array("Cint", 1e-12)
                Ro = get_array("Ro", 1e5)
                Cp = get_array("Cp", 1e-13)
                v_n = get_array("v_n", 1e-8)
                slew_rate = get_array("slew_rate", 1e7)
                swing = dict_vars.get("output_swing", 1.0)
                
                v_out_max = np.ones(order) * swing
                v_out_min = -v_out_max
                
                analog_frontend_impl = cb.GmC(
                    analog_frontend, Cint, Ro, Cp, v_n, v_out_min, v_out_max, slew_rate
                )

            # 5. Ejecutar Simulación (igual que en dataset_generator)
            snr, sndr, _, Power = analog_frontend_impl.simulateSNR(osr)
            
            max_sndr = np.nanmax(sndr) if sndr is not None else None
            max_snr = np.nanmax(snr) if snr is not None else None
            
            if max_sndr is None or np.isnan(max_sndr):
                return {"Success": False}
            
            sim_time = time.time() - start_time
            
            return {
                "SNDR": max_sndr,
                "SNR": max_snr,
                "Bw": Bw,
                "Power": Power,
                "Simulation_Time": sim_time,
                "Success": True
            }
            
        except Exception as e:
            logging.error(f"Fallo en simulador físico (Top {topology_id} - {top_name}): {e}")
            return {"Success": False}