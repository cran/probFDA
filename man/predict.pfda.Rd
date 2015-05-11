\name{predict.pfda}
\alias{predict.pfda}
\title{
Prediction method for 'pfda' class objects.
}
\description{
The prediction method for 'pfda' class objects allows to predict the labels for test observations.
}
\usage{
\method{predict}{pfda}(object, X, ...)
}
\arguments{
  \item{object}{
a supervised classifier genarated by the pfda function (a 'pfda' object).
}
  \item{X}{
the test data.
}
  \item{\dots}{
additional options for internal functions.
}
}

\value{
A list with:
\item{cls}{The predicted class labels.}
\item{P}{the posterior probabilities that observations belong to the classes.}
}
\references{
C. Bouveyron and C. Brunet, Probabilistic Fisher discriminant analysis: A robust and flexible alternative to Fisher discriminant analysis, Neurocomputing, vol. 90 (1), pp. 12-22, 2012.
}
\author{
Charles Bouveyron and Camille Brunet
}

\seealso{
\code{\link[MASS:lda]{lda}}
}
\examples{
palette(c("#E41A1C","#377EB8","#4DAF4A"))

# Simulation of data
n = 900; p = 25
n1 = 1/3*n; n2 = 1/3*n; n3 = 1/3*n; 
S1 = diag(2)
S2 = rbind(c(1,-0.95),c(-0.95,1))
S3 = rbind(c(2,0),c(0,0.05))
m1 = c(0,0); m2 = c(0,2); m3 = c(2,0)
X = rbind(mvrnorm(n1,m1,S1),mvrnorm(n2,m2,S2),mvrnorm(n3,m3,S3))
Q = qr.Q(qr(mvrnorm(p,mu=rep(0,p),Sigma=diag(25,p))))
B = mvrnorm(nrow(X),rep(0,p-2),0.1*diag(rep(p-2,p-2)))
X = crossprod(t(cbind(X,B)),Q)
cls = rep(c(1,2,3),c(n1,n2,n3))

# Cross-validation
nbrep = 10
CCR = matrix(NA,2,nbrep)
for (i in 1:nbrep){
  ind = sample(n)[1:(3/5*n)]
  lda.c = lda(X[ind,],cls[ind])
  res = predict(lda.c,X[-ind,])
  CCR[1,i] = sum(res$cl==cls[-ind])/length(cls[-ind])
  prms = pfda(X[ind,],cls[ind],model=c('DkBk','DB','AkB','AB'),crit='bic',display=TRUE)
  res = predict(prms,X[-ind,])
  CCR[2,i] = sum(res$cl==cls[-ind])/length(cls[-ind])
}

# Display results
split.screen(c(2,1))
split.screen(c(1,3), screen = 1)
screen(3)
plot(predict(princomp(X)),col=cls,pch=(17:19)[cls],main='PCA')
screen(4)
plot(crossprod(t(X),lda(X,cls)$scaling),col=cls,pch=(17:19)[cls],main='LDA')
screen(5)
plot(crossprod(t(X),pfda(X,cls,model='DkBk')$V),col=cls,pch=(17:19)[cls],main='PFDA',
  xlab='LD1',ylab='LD2')
screen(2)
boxplot(t(CCR),names=c('LDA','PFDA'),col=c(1,2),ylab="CCR",
  main='CV correct classification rate')
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ dimension reduction}
\keyword{ discriminant analysis}% __ONLY ONE__ keyword per line