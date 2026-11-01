"""Duas estações de `enlace_manual.py` num canal em memória, sem som nem cartão.

Texto nos dois sentidos em cada modulação, altura mudada, quadro perdido e
parâmetros diferentes (o que tem de virar alerta, não silêncio). Imprime
SUCCESS!/FAILED. As duas começam no mesmo instante, então a colisão também passa
por aqui. Não mede o ar.

    ./venv/bin/python simula_manual.py
"""
import time
import numpy as np, enlace_manual as M
FS=M.FS; B=2048
class Canal:
    def __init__(s,seed): s.r=np.random.default_rng(seed); s.f=[np.zeros(0,np.float32),np.zeros(0,np.float32)]
    def soma(s,i,b,d=4096):
        n=d+len(b); f=s.f[i]
        if len(f)<n: f=np.concatenate([f,np.zeros(n-len(f),np.float32)])
        f[d:d+len(b)]+=b; s.f[i]=f
    def bloco(s,i):
        x=np.zeros(B,np.float32); f=s.f[i]; k=min(B,len(f)); x[:k]=f[:k]; s.f[i]=f[k:]
        return x+(s.r.standard_normal(B)*0.002).astype(np.float32)
def sessao(pa,pb,ta,tb,lim=400,sem=1,v=False,perde=0):
    log=[[],[]]
    A=M.Estacao('chamador',semente=sem,log=log[0].append,params=pa)
    C=M.Estacao('ouvinte',semente=sem+1,log=log[1].append,params=pb)
    A.enviar(ta); C.enviar(tb); c=Canal(sem); t=0
    while t<lim*FS and not (len(C.recebido)>=len(ta) and len(A.recebido)>=len(tb) and A.confirmado and C.confirmado):
        for i,e in enumerate((A,C)):
            e.ouvir(c.bloco(i)); b=e.passo()
            if b is not None and i==0 and perde>0 and 'quadro' in b.rotulo: perde-=1; b=None
            if b is not None: c.soma(1-i,np.clip(b,-1,1).astype(np.float32)*0.5)
        t+=B
    return A,C,t/FS,log
ta=b'ola do pc, quadros de mais de uma carga aqui.\n'; tb=b'ola do celular\n'
bad=0
casos=[('16fsk',100,1),('16fsk',50,2),('16fsk',33,1),('16fsk',20,1),('8fsk',100,1),('4fsk',100,1),('2fsk',100,1),('votada',100,1),('multicanal',100,2)]
for mod,b,r in casos:
    t0=time.process_time(); p=M.Params(mod,b,r)
    A,C,t,log=sessao(p,p,ta,tb,lim=400)
    ok=bytes(C.recebido)==ta and bytes(A.recebido)==tb; bad+=not ok
    print(f"{mod:10s} {b:3d} baud rep{r}: {'OK ' if ok else 'FALHOU'} {t:6.1f}s, quadro {p.desc.segundos:.1f}s, perdidos {A.perdidos},{C.perdidos}, cpu {time.process_time()-t0:.0f}s",flush=True)
    if not ok: print(*log[0][-5:],*log[1][-5:],sep='\n')
p=M.Params('16fsk',100,1,tom_base=1000,passo_tom=120)
A,C,t,log=sessao(p,p,ta,tb); ok=bytes(C.recebido)==ta and bytes(A.recebido)==tb; bad+=not ok; print('altura 1000+120 Hz',ok,round(t,1),p.tons[-1])
A,C,t,log=sessao(M.Params(),M.Params(),ta,tb,perde=2); ok=bytes(C.recebido)==ta; bad+=not ok; print('perda de 2 quadros: entregue',ok,round(t,1),'s, sem duplicar',len(C.recebido)==len(ta))
q=M.Params(ack=False); A,C,t,log=sessao(q,q,ta,b'',lim=30,perde=1); print('sem ack, 1 quadro perdido: chegou',len(C.recebido),'de',len(ta),'bytes (esperado parcial)')
for nome,pa,pb in (('baud',M.Params('16fsk',100,1),M.Params('16fsk',50,1)),('rep',M.Params('16fsk',100,1),M.Params('16fsk',100,2)),('tons',M.Params('16fsk',100,1),M.Params('16fsk',100,1,tom_base=1100,passo_tom=130)),('mod',M.Params('16fsk',100,1),M.Params('2fsk',100,1))):
    A,C,t,log=sessao(pa,pb,ta,tb,lim=90)
    al=[l for l in log[0]+log[1] if 'ATENÇÃO' in l and 'quadro seq' not in l]
    print('divergente',nome,'recebidos',len(A.recebido),len(C.recebido),'alertas',len(al),'|',(al or ['SEM ALERTA'])[0][:150]); bad+=not al
print('SUCCESS!' if not bad else 'FAILED')
