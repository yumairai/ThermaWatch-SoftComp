def classify_status(anomaly_val):
    """
    Mengklasifikasikan status kerawanan berdasarkan nilai anomali suhu LST.
    
    Ambang Batas Default:
    - AMAN   : Anomali < 1.5 °C
    - WASPADA: 1.5 °C <= Anomali < 3.0 °C
    - BAHAYA : Anomali >= 3.0 °C
    """
    if anomaly_val is None:
        return "TIDAK TERDEFINISI"
        
    val = float(anomaly_val)
    
    if val < 1.5:
        return "AMAN"
    elif val < 3.0:
        return "WASPADA"
    else:
        return "BAHAYA"
