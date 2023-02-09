import numpy as np                                                                                                                    
from constants import *
from var_dump import var_dump
from scipy.stats import linregress
from scipy.special import gamma 
from scipy.optimize import minimize, rosen, rosen_der


def qIdI_term(vq,vI,vDI, q1,q2):
    M = np.c_[vq, vI, vDI]
    #[SMIRNOV ASK]
    #print(q1,q2)
    M_res = M[ (M[:,0]>=q1) & (M[:,0]<=q2)]
    return M_res   
      

def SFactor(q, d, ksi, r0):
    return 1.0 + d*gamma(d - 1.0) /  ( np.power(q*r0,d) * np.power( 1 + np.power(q*ksi,-2), 0.5*(d-1) ) ) *np.sin((d-1.0)* np.arctan(q*ksi) )

   

def fabs(M):
    q = M[:,0]    
    I = M[:,1]
    mq = np.sum(q)/len(q)
    

    
def make_sep(m=8):
    s=""
    for i in range(m):
        s+="="    
    return s

def print_result(variable, key=""):
    print(make_sep(pc1)+ key +make_sep(pc1)+"\n")
    var_dump(variable)
    print("\n"+make_sep(2*pc1+len(key))+"\n")


def monitor_matrix(M, matrix_name):
    print("***************************************************"+matrix_name+"***************************************************")
    if M.ndim>1:
        rows,cols = np.shape(M)
        if rows > 0 and cols > 0:
            first_row = M[0]
            last_row= M[rows-1]

            print("Columns="+str(cols))    
    else:
        rows = len(M)
        first_row = M[0]
        last_row= M[rows-1]
    print("Rows="+str(rows))    
    print("FIRST_ROW="+str(first_row))    
    print("LAST_ROW="+str(last_row))    
    print("*******************************************************************************************************************\n")


def get_fi(x, a , b , d, ksi, r0):
    return a + b*np.power(x,2) + np.log(SFactor(x, d, ksi,r0))

def get_delta(Intens, x , a , b , d, ksi, r0):
    return np.sum( np.power(get_fi(x , a, b , d, ksi, r0)- np.log(Intens),2 ))

def get_delta_for_min(params, Intens, x ):
    a , b , d, ksi, r0 = params
    return np.sum( np.power(get_fi(x , a, b , d, ksi, r0)- np.log(Intens),2 ))

def get_MN(Intens, SF):
    return Intens[0] / SF   
 
def min_get_delta(I,q,params):
    a                                 = params[0]
    b                         = params[1]
    d                 = params[2]
    ksi       = params[3]
    r0= params[4]
    res = minimize( get_delta_for_min, [a,b,d,ksi,r0], method='BFGS',args=(I,q) , tol=1e-6, 
options={
#'xatol': 1e-11,
 'disp': True})
    print(res)
    return res

def get_r0(q1):
    return 2*np.pi / q1 

def get_KSI(r0):
    return 10*r0

#Guinie slope
def get_B():
    return 0.0

def get_A(mn):
    return np.log(mn)


def get_D( q, I,cut_of = 0.1):
    M = np.c_[q, I]

    print(np.shape(M))
    M_res = M[ (M[:,0]<=cut_of)]
    print(np.shape(M_res))
    lr = linregress( np.power(M_res[:,0],2), np.log(M_res[:,1]))
    print(lr)
    return lr.slope   


def plot_SRC_AND_Result(plt, qsrc,isrc, ires, sf, guinier=False):
    plt.clf()
    plt.close()


    if guinier:
        plt.plot(np.power(qsrc,2), np.log(isrc),  ms=2, marker="_",color="b")
        plt.plot(np.power(qsrc,2),np.log(ires), ms=1, marker="_",color="r")
        plt.plot(np.power(qsrc,2), np.log(ires*sf), ms=3, marker="o", color="g")
    else:
        plt.plot(qsrc, isrc,  ms=2, marker="_",color="b")
        plt.plot(qsrc,ires, ms=1, marker="_",color="r")
        plt.plot(qsrc, ires*sf, ms=3, marker="o", color="g")

    plt.loglog()
    plt.show()


def plot_two_single_func(plt, arg1,fn1,arg2,fn2):
    plt.clf()
    plt.plot(arg1, fn1,  ms=3, marker="_",color="r")
    plt.plot(arg2, fn2,  ms=4, marker="o", color="g")
    plt.show()




def plot_multiple_func(plt, axis = ("q, nm^-1","I, arb. units") ,guinier=False,isDoubleLog=False, erraze=True, *args):
    
    if erraze:
        plt.clf()
        plt.close()

    if len(args) == 2 :

        print("Two vars func")
        if guinier == False :
            print("That is Not Gunier branch")
            x = [ el[0] for el in args]
            y =[ el[1] for el in args] 
            lbl = [ el[2] for el in args]  
        else:
            print("This is Not Gunier branch")
            print_result(args,"args")
            x = [np.power(el[0],2) for el in args]
            y = [np.log(el[1]) for el in args]          
            lbl = [el[2] for el in args]   
            #plt.title = "Ln(I)(q^2)"               
        plt.plot(x[0], y[0],  ms=2, marker="o",color="b",label=lbl[0])
        plt.plot(x[1], y[1],  ms=2, marker="_",color="g",label=lbl[1])
  

    if len(args) == 3 :
        if guinier == False :
            x = [ el[0] for el in args]
            y =[ el[1] for el in args] 
            lbl = [ el[2] for el in args]  
            
        else:
            x = [np.power(el[0],2) for el in args]
            y = [np.log(el[1]) for el in args]          
            lbl = [el[2] for el in args]                  
        plt.plot(x[0], y[0],  ms=3, marker="o",color="r",label=lbl[0])
        plt.plot(x[1], y[1],  ms=2, marker="_",color="y",label=lbl[1])
        plt.plot(x[2], y[2],  ms=2, marker="_",color="g",label=lbl[2])

 

    if len(args) == 4:    
        if guinier == False :
            x = [ el[0] for el in args]
            y =[ el[1] for el in args] 
            lbl = [ el[2] for el in args]  
        else:
            x = [np.power(el[0],2) for el in args]
            y = [np.log(el[1]) for el in args]          
            lbl = [el[2] for el in args]                  


        plt.plot(x[0], y[0],  ms=3, marker="_",color="r",label=lbl[0])
        plt.plot(x[1], y[1],  ms=3, marker="_",color="r",label=lbl[1])
        plt.plot(x[2], y[2],  ms=3, marker="_",color="r",label=lbl[2])
        plt.plot(x[3], y[3],  ms=4, marker="o", color="g",label=lbl[3])

    plt.xlabel(lbl[0])
    plt.ylabel(lbl[1])

    if isDoubleLog:
        plt.loglog()
    plt.legend()

    plt.xlabel(f'${axis[0]}$')
    plt.ylabel(f'${axis[1]}$')
    plt.show()



