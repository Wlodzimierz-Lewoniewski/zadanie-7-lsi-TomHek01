import numpy as np
import re

def przetworz_tekst(txt):
    
    return re.sub(r'[^\w\s]', '', txt.lower())

def zbuduj_macierz(dok, termy, indeks):
    
    macierz = np.zeros((len(termy), len(dok)))
    for j, d in enumerate(dok):
        for s in przetworz_tekst(d).split():
            if s in indeks:
                macierz[indeks[s], j] = 1
    return macierz

def zredukuj_macierz(macierz, k):
    
    U, S, Vt = np.linalg.svd(macierz, full_matrices=False)
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    V_k = Vt[:k, :]
    return S_k, U_k, V_k

def oblicz_podobienstwa(dok_redukcja, zap_redukcja):
    
    podobienstwa = []
    for i in range(dok_redukcja.shape[1]):
        if np.linalg.norm(dok_redukcja[:, i]) != 0:
            podobienstwo = float(
                np.dot(zap_redukcja, dok_redukcja[:, i]) /
                (np.linalg.norm(zap_redukcja) * np.linalg.norm(dok_redukcja[:, i]))
            )
        else:
            podobienstwo = 0
        podobienstwa.append(podobienstwo)
    return podobienstwa

def lsi(dok, zap, k):
    
    przetworzone_dok = [przetworz_tekst(d) for d in dok]
    termy = sorted(set(term for d in przetworzone_dok for term in d.split()))
    indeks = {t: i for i, t in enumerate(termy)}
    
    macierz = zbuduj_macierz(dok, termy, indeks)
    S_k, U_k, V_k = zredukuj_macierz(macierz, k)
    
    dok_redukcja = np.dot(S_k, V_k)
    
    v_zap = np.zeros(len(termy))
    for s in przetworz_tekst(zap).split():
        if s in indeks:
            v_zap[indeks[s]] = 1
    
    zap_redukcja = np.dot(np.dot(v_zap, U_k), np.linalg.inv(S_k))
    podobienstwa = oblicz_podobienstwa(dok_redukcja, zap_redukcja)
    
    return [round(p, 2) for p in podobienstwa]

def main():
    n = int(input())
    dok = [input() for i in range(n)]
    zap = input()
    k = int(input())
    
    wynik = lsi(dok, zap, k)
    print(wynik)

if __name__ == "__main__":
    main()
