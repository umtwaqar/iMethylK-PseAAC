import math
import numpy
from sklearn.preprocessing import StandardScaler
from webApp import modelLoader as mL


class extractFeatures():

    def seqToMat(seq):
        encoder = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                   'Y']
        len = seq.__len__()
        n = int (math.ceil(math.sqrt(len)))
        seqMat = [[0 for x in range(n)] for y in range(n)]
        i=0
        seqiter=0
        for i in range(n):
            j=0
            for j in range(n):
                if seqiter < len:
                    try:
                        aa=int(encoder.index(seq[seqiter]))
                    except ValueError:
                        exit(0)
                    else:
                        seqMat[i][j]= aa
                    seqiter+=1
        return seqMat

    def frequencyVec(seq):
        encoder = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                   'Y']
        fv = [0 for x in range(21)]
        i = 1
        for i in range(21):
            fv[i-1] = seq.count(encoder[i])
        fv[20] = seq.count('X')
        return fv

    def AAPIV(seq):
        encoder = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                   'Y']
        apv = [0 for x in range(21)]
        i=1
        sum = 0
        for i in range(21):
            j=0
            for j in range(len(seq)):
                if seq[j] == encoder[i]:
                    sum=sum+j+1
            apv[i] = sum
            sum=0
        return apv[1:]+apv[0:1]

    def SVV(seq):
        encoder = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                   'Y']
        svv = [0 for x in range(len(seq))]
        i = 0
        for i in range(len(seq)):
            value = encoder.index(seq[i])
            if value == 0:
                svv[i] = 21
            else:
                svv[i] = value
        return svv

    def print2Dmat(mat):
        n =len(mat)
        i=0
        strOut=''
        for i in range(n):
            strOut = strOut+str(mat[i])+'<br>'
        return strOut

    def PRIM(seq):
        encoder = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                   'Y', 'X']
        prim = [[0 for x in range(21)] for y in range(21)]
        i=0
        for i in range(21):
            aa1=encoder[i]
            aa1index=-1
            for x in range(len(seq)):
                if seq[x] == aa1:
                    aa1index=x+1
                    break
            if aa1index!=-1:
                j=0
                for j in range (21):
                    if j!=i:
                        aa2=encoder[j]
                        aa2index = 0
                        for y in range(len(seq)):
                            if seq[y] == aa2:
                                aa2index = aa2index+((y + 1)-aa1index)
                        prim[i][j] = int(aa2index)
        return prim

    def rawMoments(mat,order):
        n = len(mat)
        rawM = []
        sum=0
        i=0
        for i in range(order+1):
            j=0
            for j in range(order+1):
                if i+j <= order:
                    p=0
                    for p in range(n):
                        q=0
                        for q in range(n):
                            sum=sum+(((p+1)**i)*((q+1)**j)*int(mat[p][q]))
                    rawM.append(sum)
                    sum=0
        return rawM

    def centralMoments(mat,order, xbar, ybar):
        n = len(mat)
        centM = []
        sum=0
        i=0
        for i in range(order+1):
            j=0
            for j in range(order+1):
                if i+j <= order:
                    p=0
                    for p in range(n):
                        q=0
                        for q in range(n):
                            sum=sum+((((p+1)-xbar)**i)*(((q+1)-ybar)**j)*mat[p][q])
                    centM.append(sum)
                    sum=0
        return centM

    def hahnMoments(mat,order):
        N = len(mat)
        hahnM = []
        i = 0
        for i in range(order+1):
            j = 0
            for j in range(order+1):
                if i+j <= order:
                    answer = extractFeatures.hahnMoment(i,j,N,mat)
                    hahnM.append(answer)
        return hahnM

    def hahnMoment(m,n,N,mat):
        value = 0.0
        x=0
        for x in range(N):
            y=0
            for y in range(N):
                value = value + (mat[x][y]*(extractFeatures.hahnProcessor(x,m,N))*(extractFeatures.hahnProcessor(x,n,N)))
        return value

    def hahnProcessor(x,n,N):
        return extractFeatures.hahnPol(x,n,N)*math.sqrt(extractFeatures.roho(x,n,N))

    def hahnPol(x,n,N):
        answer =0.0
        ans1 = extractFeatures.pochHammer(N-1.0,n) * extractFeatures.pochHammer(N-1.0,n)
        ans2 = 0.0
        k=0
        for k in range(n+1):
            ans2 = ans2+math.pow(-1.0,k) * ((extractFeatures.pochHammer(-n,k) * extractFeatures.pochHammer(-x,k) *
                                             extractFeatures.pochHammer(2*N-n-1.0,k)))
        answer = ans1 + ans2
        return answer

    def roho(x,n,N):
        return extractFeatures.gamma(n+1.0) * extractFeatures.gamma(n+1.0) *extractFeatures.pochHammer((n+1.0),N)

    def gamma(x):
        return math.exp(extractFeatures.logGamma(x))

    def logGamma(x):
        temp = (x-0.5)*math.log(x+4.5)-(x+4.5)
        ser = 101.19539853003
        return temp + math.log(ser * math.sqrt(2*math.pi))

    def pochHammer(a,k):
        answer = 1.0
        i = 0
        for i in range(k):
            answer=answer*(a+i)
        return answer

    def calcFV(seq):
        fv = [0 for x in range(194)]
        fvIter =0
        myMat = extractFeatures.seqToMat(seq)
        myRawMoments = extractFeatures.rawMoments(myMat,3)
        for ele in myRawMoments:
            fv[fvIter] = ele
            fvIter= fvIter+1
        xbar = myRawMoments[4]
        ybar = myRawMoments[1]
        myCentralMoments = extractFeatures.centralMoments(myMat,3,xbar,ybar)
        for ele in myCentralMoments:
            fv[fvIter] = ele
            fvIter= fvIter+1
        myHahnMoments = extractFeatures.hahnMoments(myMat, 3)
        for ele in myHahnMoments:
            fv[fvIter] = ele
            fvIter= fvIter+1
        myFrequencyVec = extractFeatures.frequencyVec(seq)
        for ele in myFrequencyVec:
            fv[fvIter] = ele
            fvIter= fvIter+1
        mySiteVicinityVec = extractFeatures.SVV(seq)
        for ele in mySiteVicinityVec:
             fv[fvIter] = ele
             fvIter= fvIter+1
        myPRIM = extractFeatures.PRIM(seq)
        myPRIMRawMoments = extractFeatures.rawMoments(myPRIM,3)
        xbar2 = myPRIMRawMoments[4]
        ybar2 = myPRIMRawMoments[1]
        myPRIMCentralMoments = extractFeatures.centralMoments(myPRIM,3,xbar2,ybar2)
        for ele in myPRIMRawMoments:
            fv[fvIter] = ele
            fvIter= fvIter+1
        for ele in myPRIMCentralMoments:
            fv[fvIter] = ele
            fvIter= fvIter+1
        myPRIMHahnMoments = extractFeatures.hahnMoments(myPRIM, 3)
        for ele in myPRIMHahnMoments:
            fv[fvIter] = ele
            fvIter= fvIter+1
        myAAPIV = extractFeatures.AAPIV(seq)
        for ele in myAAPIV:
            fv[fvIter] = ele
            fvIter= fvIter+1
        myRPRIM = extractFeatures.PRIM(seq[::-1])
        myRPRIMRawMoments = extractFeatures.rawMoments(myRPRIM, 3)
        xbar3 = myRPRIMRawMoments[4]
        ybar3 = myRPRIMRawMoments[1]
        myRPRIMCentralMoments = extractFeatures.centralMoments(myRPRIM, 3, xbar3, ybar3)
        for ele in myRPRIMRawMoments:
            fv[fvIter] = ele
            fvIter= fvIter+1
        for ele in myRPRIMCentralMoments:
            fv[fvIter] = ele
            fvIter= fvIter+1
        myRPRIMHahnMoments = extractFeatures.hahnMoments(myRPRIM, 3)
        for ele in myRPRIMHahnMoments:
            fv[fvIter] = ele
            fvIter= fvIter+1
        myRAAPIV = extractFeatures.AAPIV(seq[::-1])
        for ele in myRAAPIV:
            fv[fvIter] = ele
            fvIter= fvIter+1       
        fv = numpy.asarray(fv).reshape(1,-1)
        std_scale = StandardScaler().fit(fv)
        scale, mean = mL.modelLoader.loadScaler(mL)
        std_scale.mean_ = mean
        std_scale.scale_ = scale
        newfv = std_scale.transform(fv)
        return newfv

    def processAllStrings(allSeq):
        seqs = str(allSeq).split('-')
        outTable = ""
        myseq = []
        allowed_chars=set('ACDEFGHIKLMNPQRSTVWXY')
        i=0
        for seq in seqs:
            if seq!='':
                if set(seq).issubset(allowed_chars):
                    myseq = list(seq)
                    myseq[20] = '<b style="color:red">K</b>'
                    myseq = "".join(myseq)
                    outTable = outTable + myseq + '  |   ' + \
                              str(extractFeatures.performPrediction(
                                  numpy.asarray(extractFeatures.calcFV(seq)).reshape(1, 194))) \
                              + '<br>'
                    i=i+1
                    myseq = []
                else:
                    outTable = outTable + '<b>******************************Sequence ' + str(i) + \
                             '******************************</b><br>'+seq+'<br>Invalid Sequence <br>'
                    i = i + 1

        return outTable

    def driverPredictor(allSeqData):
        samples=''
        seqs = str(allSeqData).split()
        for seq in seqs:
            seqNew = 'XXXXXXXXXXXXXXXXXXXX'+seq+'XXXXXXXXXXXXXXXXXXXX'
            for i in range(len(seqNew)):
                if seqNew[i]=='K':
                    samples=samples+seqNew[i-20:i+21]+'-'
        if samples!='':
            ans = extractFeatures.processAllStrings(samples)
        else:
            ans='Sequence does not contains Aspartate'
        return ans

    def performPrediction(FV):
        model = mL.modelLoader.loadModel(mL)
        output = model.predict(FV)
        y1=output[0]
        yLabel=numpy.ndarray.round(y1)
        if yLabel==1:
            return '<b style="background:#00FF7F;">Methyl Lysine</b>'
        else:
            return '<i style="background:#e271a5">Non Methyl Lysine</i>'