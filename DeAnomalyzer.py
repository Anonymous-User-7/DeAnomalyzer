import sys
import subprocess

fname = str(sys.argv[1])

if __name__ == '__main__':
    print("DeAnomalyzer\n\n")
    print("Select one of the following action by entering the serial number")
    print("\t1. Reduce Non-determinism in Isolation Forest: Scikit-Learn")
    print("\t2. Reduce Non-determinism in Isolation Forest: Matlab")
    print("\t3. Reduce Non-determinism in Isolation Forest: R")
    print("\t4. Reduce Non-determinism in Robust Covariance: Scikit-Learn")
    print("\t5. Reduce Non-determinism in Robust Covariance: Matlab")
    print("\t6. Reduce Non-determinism in One Class SVM: Matlab")
    print()
    print("\t7. Reduce Inconsistency in Isolation Forest")
    print("\t8. Reduce Inconsistency in Robust Covariance")
    print("\t9. Reduce Inconsistency in Local Outlier Factor")
    print("\t10. Reduce Inconsistency in One Class SVM")
    print("Press Enter after completion")
    
    n = int(input("Choice: "))
    if n==1:
        subprocess.Popen(["python", "SkIF_GD.py", fname])
    elif n ==2:
        subprocess.Popen(["python", "MatIF_GD.py", fname])
    elif n ==3:
        subprocess.Popen(["python", "RIF_GD.py", fname])
    elif n ==4:
        subprocess.Popen(["python", "SkEE_GD.py", fname])
    elif n ==5:
        subprocess.Popen(["python", "MatEE_GD.py", fname])
    elif n ==6:
        subprocess.Popen(["python", "MatOCSVM_GD.py", fname])
        
    elif n ==7:
        subprocess.Popen(["python", "Inconsistency_IF.py", fname])
    elif n ==8:
        subprocess.Popen(["python", "Inconsistency_EE.py", fname])
    elif n ==9:
        subprocess.Popen(["python", "Inconsistency_LOF.py", fname])
    elif n ==10:
        subprocess.Popen(["python", "Inconsistency_OCSVM.py", fname])
    else:
        print("Try Again")
        
    