# Subset of FEBA.R to create PDFs of plots


FEBA_Make_PDFs = function(args = commandArgs(trailingOnly=TRUE)) {
    path_to_tables_dir = args[1];
    op_dir = args[2];

    print(path_to_tables_dir);
    stop("9");

    fit = getfit(path_to_tables_dir)
    genes = getgenes(path_to_tables_dir)
     
    
    nameToPath = function(filename) paste(op_dir, filename,sep="/");
    wroteName = function(x) cat("Wrote ",nameToPath(x),"\n",file=stderr());
    
    
    #FEBA_Save_Tables(fit', 'genes, org, expsU=exps, dir=dir, FEBAdir=FEBAdir);
    
    
    FEBA_Quality_Plot(fit$q, nameToPath("fit_quality.pdf"), org, ...);
    wroteName("fit_quality.pdf");
    
    if(is.null(fit$pairs)) {
    	paste("No data for cofitness plot\n");
    	unlink(nameToPath("cofitness.pdf"));
    } else {
    	FEBA_Cofitness_Plot(fit$pairs, nameToPath("cofitness.pdf"), org);
    	wroteName("cofitness.pdf");
    }
    
    pdf(nameToPath("fit_quality_cor12.pdf"),
    	pointsize=10, width=6, height=6,
    	title=paste(org,"Fitness Cor12 Plots"));
    for (i in 1:nrow(fit$q)) {
        n = as.character(fit$q$name[i]);
        changers = fit$g[abs(fit$t[[n]]) >= 3];
        plot(fit$lrn1[[n]], fit$lrn2[[n]],
        		  main=sprintf("%s %s #%d (gMed=%.0f rho12=%.3f)\n%s",
    		  	org, n, fit$q$num[i], fit$q$gMed[i], fit$q$cor12[i], fit$q$short[i]),
        		  xlab="First Half", ylab="Second Half",
    		  col=ifelse(fit$g %in% changers, 2, 1));
        eqline(); hline(0); vline(0);
    }
    dev.off();
    wroteName("fit_quality_cor12.pdf");
    
    labelAll = sprintf("%s #%d gMed=%.0f rho12=%.2f %30.30s",
    	      sub("^set","",fit$q$name), fit$q$num, fit$q$gMed, fit$q$cor12, fit$q$short);
    labelAll = ifelse(fit$q$short=="Time0", paste(labelAll, fit$q$t0set), labelAll);
    
    use = fit$q$short != "Time0";
    if(sum(use) > 2) {
        lrClust = hclust(as.dist(1-cor(fit$lrn[,as.character(fit$q$name)[use]], use="p")));
        pdf(nameToPath("fit_cluster_logratios.pdf"),
    	pointsize=8, width=0.25*pmax(8,sum(use)), height=8,
    	title=paste(org,"Cluster Logratios"));
        plot(lrClust, labels=labelAll[use], main="");
        dev.off();
        wroteName("fit_cluster_logratios.pdf");
    }
    
    if (ncol(fit$gN)-1 >= 3) { # at least 3 things to cluster
        countClust = hclust(as.dist(1-cor(log2(1+fit$gN[fit$gN$locusId %in% fit$genesUsed,-1]))));
        pdf(nameToPath("fit_cluster_logcounts.pdf"),
    	pointsize=8, width=pmax(5,0.25*nrow(fit$q)), height=8,
    	title=paste(org,"Cluster Log Counts"));
        # Some Time0s may be missing from fit$q
        d = match(names(fit$gN)[-1], fit$q$name);
        labelAll2 = ifelse(is.na(d), paste("Time0", sub("^set","",names(fit$gN)[-1])), labelAll[d]);
        plot(countClust, labels=labelAll2, main="");
        dev.off();
        wroteName("fit_cluster_logcounts.pdf");
    }
    
    d = table(genes$scaffoldId[genes$locusId %in% fit$genesUsed]);
    maxSc = names(d)[which.max(d)];
    if (is.null(maxSc)) stop("Invalid scaffoldId?");
    beg = ifelse(fit$g %in% genes$locusId[genes$scaffold==maxSc],
        genes$begin[match(fit$g, genes$locusId)], NA);
    
    pdf(nameToPath("fit_chr_bias.pdf"), pointsize=10, width=6, height=6,
              title=paste(org,"Chromosome Bias"));
    for (i in 1:nrow(fit$q)) {
        n = as.character(fit$q$name[i]);
        plot(beg, pmax(-2,pmin(2,fit$lr[[n]])),
        		  main=sprintf("%s %s #%d (gMed=%.0f rho12=%.3f)\n%s",
    		  	org, sub("^set","",n), fit$q$num[i], fit$q$gMed[i], fit$q$cor12[i], fit$q$short[i]),
        		  xlab="Position on Main Scaffold",
    		  ylab="Fitness (Unnormalized)",
    		  ylim=c(-2,2), col="darkgrey");
        o = order(beg);
        lines(beg[o], (fit$lr[[n]] - fit$lrn[[n]])[o], col="darkgreen", lwd=2);
        hline(0,lty=1,col=1);
    }
    dev.off();
    wroteName("fit_chr_bias.pdf");

}


FEBA_Quality_Plot = function(q, pdfFile, org,
		             min_gMed=50, max_mad12=0.5, max_adjcor = 0.25, max_gccor = 0.2, min_cor12 = 0.1,
			     multiples=TRUE) {
	qCol = ifelse(q$short=="Time0", "grey",
		ifelse(q$u, ifelse(q$maxFit > 5, "blue", "darkgreen"), "red"));
	qLab = q$num;

	if(!is.null(pdfFile)) pdf(pdfFile,
		pointsize=10, width=8, height=8,
		title=paste(org,"Fitness Quality"));
	oldpar = par(mfrow=c(2,2));

	# mad12 vs. gMed
	plotlab(0.1 + q$gMed, pmin(0.75,q$mad12), qLab, col=qCol, ylim=c(0,0.75), log="x",
		            cex=0.8,
			    xlab="Median Reads per Gene",
			    ylab="Median abs. diff.(1st half, 2nd half)",
			    main=paste(org,"mad12"));
	vline(min_gMed); hline(max_mad12);

	# cor12 vs. mad12
	plotlab(pmin(0.75,q$mad12), pmax(0,q$cor12), qLab, col=qCol, xlim=c(0,0.75), ylim=0:1,
			     cex=0.8,
			     xlab="Median abs. diff.(1st half, 2nd half)", ylab="rho(1st half, 2nd half)",
			     main=paste(org,"rho12"));
	vline(max_mad12); hline(min_cor12);

	# opcor vs. adjcor
	plotlab(abs(q$adjcor), pmax(0,q$opcor), qLab, col=qCol, xlim=0:1, ylim=0:1,
			     cex=0.8,
			     xlab="| rho(adj. genes on diff. strands) |",
			     ylab="rho(operon pairs)",
			     main=paste(org,"rho(operons)"));
	eqline();
	vline(max_adjcor);

	# adjcor vs. gccor
	plotlab(abs(q$gccor), abs(q$adjcor), qLab, col=qCol, xlim=0:1, ylim=0:1,
			     cex=0.8,
			     xlab="| cor(gene GC, fitness) |",
			     ylab="rho(adj. genes on diff. strands)",
			     main=paste(org,"GC effects"));
	eqline();
	vline(max_gccor); hline(max_adjcor);

	if (!is.null(pdfFile) && multiples) {
	    for(t0set in unique(q$t0set)) {
	        FEBA_Quality_Plot(q[q$t0set==t0set,], pdfFile=NULL, org=t0set, multiples=FALSE);
	    }
	}
	par(oldpar);
	if(!is.null(pdfFile)) dev.off();
}




FEBA_Cofitness_Plot = function(pairs, pdfFile, org) {
	if(!is.null(pdfFile)) pdf(pdfFile,
		pointsize=10, width=4, height=4,
		title=paste(org,"Cofitness"));
	CompareDensities(list(Operon=withoutNA(pairs$pred$rfit[pairs$pred$bOp]),
			          Adjacent=withoutNA(pairs$adjDiff$rfit),
				  Random=withoutNA(pairs$random$rfit)),
			 legendX="topleft",
			 xlim=c(-1,1),
			 xlab="Cofitness", ylab="Density", lwd=c(1,2,2), col=c(3,2,1), lty=c(1,2,4),
			 main=paste("Cofitness in",org));
	if(!is.null(pdfFile)) dev.off();
}



# Just like plot.default() but adds on the labels
# By default plotting symbols are off but you can override that
plotlab = function(x,y,labels, cex=1, col=1, pch="", ...) {
	plot.default(x,y,cex=cex,col=col,pch=pch,...);
	text(x,y,labels,cex=cex,col=col);
}

### Graphics utilities
hline <- function(y,col="grey",lty=2,lwd=1) {
	lines(c(-1e20,1e-40,1e20),c(y,y,y),col=col,lty=lty,lwd=lwd);
}

vline <- function(x,col="grey",lty=2,lwd=1) {
	lines(c(x,x,x),c(-1e20,1e-40,1e20),col=col,lty=lty,lwd=lwd);
}

eqline <- function(col="grey",lty=2,lwd=1) {
	x <- 10**(-25:25);
	lines(c(-rev(x),x),c(-rev(x),x),col=col,lty=lty,lwd=lwd);
}



CompareDensities <- function(list,labels=names(list),xlim=range(unlist(list)),ylim=c(0,3),
		col=1:length(labels), lty=1:length(labels), lwd=rep(1,length(labels)),
		legendX=mean(xlim),legendY=ylim[2],main="",xlab="",ylab="",showCounts=FALSE,
		showLegend=TRUE, bty="o") {

	for (i in 1:length(labels)) {
		x = list[[ names(list)[i] ]];
		d <- density(x,from=xlim[1],to=xlim[2]);
		if(i==1) {
			plot(d$x,d$y,type="l",col=col[i],lty=lty[i],lwd=lwd[i],
				main=main,xlab=xlab,ylab=ylab,xlim=xlim,ylim=ylim);
		} else {
			lines(d$x,d$y,col=col[i],lty=lty[i],lwd=lwd[i]);
		}
		if (showCounts) labels[i] <- paste(labels[i]," (n=", sum(!is.na(x)), ")", sep="");
	}
	if(showLegend) {
	       if(is.numeric(legendX)) {
	           legend(legendX,legendY,labels,col=col,lty=lty,lwd=lwd, bty=bty);
	       } else { # e.g. "topleft"
	           legend(legendX, labels,col=col,lty=lty,lwd=lwd, bty=bty);
 	       }
	}
}



getfit = function(path_to_tables_dir) {
    # What is contained in fit?
    # fit:
    #    pairs
    #       pred
    #       adjDiff
    #       random
    #    q (Quality DataFrame)
    #    t
    #    lr
    #    lrn
    #    lrn1
    #    lrn2
    #    gN
    #    genesUsed
    #    g (locusIds)
    ptd = path_to_tables_dir;
   
    # Creating pred
    prd = read.table(paste(ptd,"pred.tsv",sep=""),as.is=T);
    ajD = read.table(paste(ptd,"adjDiff.tsv",sep=""),as.is=T);
    rdm = read.table(paste(ptd,"random.tsv",sep=""),as.is=T);
    pred = list(pred=prd,adjDiff=ajD,random=rdm)

    q_df = read.table(paste(ptd,"q.tsv",sep=""),as.is=T);
    t_df = read.table(paste(ptd,"t.tsv",sep=""),as.is=T);
    lr_df = read.table(paste(ptd,"lr.tsv",sep=""),as.is=T);
    lrn_df = read.table(paste(ptd,"lrn.tsv",sep=""),as.is=T);
    lrn1_df = read.table(paste(ptd,"lrn1.tsv",sep=""),as.is=T);
    lrn2_df = read.table(paste(ptd,"lrn2.tsv",sep=""),as.is=T);
    gN_df = read.table(paste(ptd,"gN.tsv",sep=""),as.is=T);


    # Read two lists (?)
    g = scan(paste(ptd, "g.tsv", sep=""), as.is=T);
    genesUsed = scan(paste(ptd, "genesUsed.tsv", sep=""), as.is=T);



}


# Actually do the work
if(!interactive()) {
	FEBA_Make_PDFs();
	quit();
}
