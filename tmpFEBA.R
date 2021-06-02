HighFit = function(fit, genes, expsUsed, all,
                                      min.fit=4, min.t=5, max.se=2, min.reads=10, min.gMean=10, max.below=8,
                                                         min.strains=2, min.strain.fraction=0.5) {
      wHigh = which(fit$lrn >= min.fit & fit$t >= min.t & fit$tot >= min.reads & fit$n >= min.strains, arr.ind=T);
  high = data.frame(locusId=fit$g[wHigh[,1]], expName=names(fit$lrn)[wHigh[,2]],
                                        fit=fit$lrn[wHigh], t=fit$t[wHigh], nReads=fit$tot[wHigh], nStrains=fit$n[wHigh]);
    # t ~= fit/standard_error, so estimate s.e. = fit/t
    high$se = high$fit/high$t;
      high$sdNaive = fit$sdNaive[wHigh];
      high = subset(high, se <= max.se);

        # which experiments are ok
        fields = words("name Group Condition_1 Concentration_1 Units_1 Media short");
        fields = fields[fields %in% names(expsUsed)];
          exps = expsUsed[, fields];
          exps = merge(exps, fit$q[,words("name u short maxFit gMean")]);
            high = merge(high, exps, by.x="expName", by.y="name");
            high = subset(high, gMean >= min.gMean & fit >= maxFit - max.below);
              names(high)[names(high)=="u"] = "used";
              high = merge(genes[,c("locusId","sysName","desc")], high);

    # Compute #strains detected per gene x sample
    u = all$locusId %in% high$locusId & fit$strainsUsed;
    d = all[u, names(all) %in% high$expName];
    nDetected = aggregate(d > 0, all[u,"locusId",drop=F], sum);
    expNames = names(nDetected)[-1];
    nDetected = data.frame(locusId=rep(nDetected$locusId, length(expNames)),
    expName=rep(expNames, each=nrow(nDetected)),
    # values, by experiment
    nDetected = unlist(nDetected[,-1]));
    high = merge(high, nDetected);
    high = subset(high, nDetected/nStrains >= min.strain.fraction);
    high = high[order(high$expName, -high$fit),];
                        return(high);
