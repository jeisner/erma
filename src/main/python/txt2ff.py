#!/usr/bin/env python

## Convert data from CoNLL200 to ERMA format 

import sys
import re
import os.path
import time, subprocess, random
from optparse import OptionParser

#Output only NP chunks (all other chunks become O)
npsonly = False
#POS tags are given as an input
inpostags=True
#All words with frequency less than the limit will be tagged as out of vocabulary (OOV)
freq_limit=1
num_examples = 20000
feat_context=0
word_context=2
oovs={}
#featrange = range(-feat_context,feat_context+1)
wordrange = range(-word_context,word_context+1)
featrange = wordrange

usage = "usage: %prog [options] datafile"
parser = OptionParser(usage=usage)
parser.add_option("-n", "--npsonly",
                  dest="npsonly",default=False,action="store_true" )
parser.add_option("-t", "--printpos", default=True,
                  dest="printpos", action="store_false",
                  help="set this option to disregard postags")

(options, args) = parser.parse_args()
if len(args) != 1:
        parser.error("incorrect number of arguments")
printpostags = options.printpos
npsonly = options.npsonly

print "npsonly="+str(npsonly)+";\tprint POS="+str(printpostags)+" inPosTags="+str(inpostags)

data_file = args[0]

brilltags={}
punc = re.compile("\W")

#Regular expressions that express some features of words
#Is the word capitalized?
capitalized = re.compile("[A-Z][a-z]+")
#Is the word a single capital letter?
caplet = re.compile("[A-Z]$")
#The word consists only of capital letters
allcaps = re.compile("[A-Z]+$")
#Tries to capture e.g. McLovin
caps = re.compile("[A-Z]+[a-z]+[A-Z]+[a-z]")
#The word contains a number
contains_num = re.compile("\d+")
#The word contains a hyphen
contains_hyp = re.compile("\-")
#The word contains an appostophy
contains_appos = re.compile("\'")
#The word contains a period
contains_per = re.compile("[A-Za-z]\.$")
#The word is a number
is_num = re.compile("[+-]?\d+(\,\d+)?\.?\d*$")
#The word is an integer number
is_int = re.compile("\d+$")
#The word ends in s
ends_s = re.compile("[A-Za-z]+s$")
#The word ends in ing
ends_ing = re.compile("[A-Za-z]+ing$")
#The word ends in ed
ends_ed = re.compile("[A-Za-z]+ed$")

if inpostags:
	feat_suf = ["_chunk(CHUNK):=[*]"]
else:
	feat_suf = ["_pos(POS):=[*]","_chunk(CHUNK):=[*]"]

feat_suf.append("_chunk_link(CHUNK,CHUNK):=[*,*]")

string_map = {".":"_P_" , ",":"_C_", "'":"_A_", "%":"_PCT_", "-":"_DASH_",
	      "$":"_DOL_", "&":"_AMP_", ":":"_COL_", ";":"_SCOL_", "\\":"_BSL_"
	      , "/":"_SL_", "`":"_QT_", "?":"_Q_", "=":"_EQ_", "*":"_ST_",
	      "!":"_E_", "#":"_HSH_", "@":"_AT_", "(":"_LBR_", ")":"_RBR_"
	      , "\"":"_QT1_"}
def read_list(filename):
	result = {}
	inf = open(filename)
	for line in inf:
		line = line.strip()
		w = line.split()[0]
		if w!="": result[w]=1
	return result

#names = read_list("lists/names.lst")
#titles = read_list("lists/title.lst")
#temporals = read_list("lists/temporal_words.lst")
#geos = read_list("lists/geo.lst")
other_features=["CAPITALIZED","CAPLET","ALLCAPS","CAPS","CONTAINS_NUM",
		"CONTAINS_HYP","CONTAINS_APPOS","CONTAINS_PER","IS_NUM",
		"IS_INT","ENDS_S","ENDS_ED","ENDS_ING"]

def escape(s):
	for val, repl in string_map.iteritems():
		s = s.replace(val,repl)
	return s
	
def clean(s):
	s = escape(s)
	if re.match("[^A-Za-z0-9_]+",s):
		print s
	return punc.sub("_",s)

def printind(s):
	return str(-s).replace("-","m")

def norm_digits(s):
	return re.sub("\d","0",s)

#Output the features for a given example
def printfeats(s,ind,maxind):
	result = ""
	for i in featrange:
		index = ind+i
		if index>=0 and index<=maxind:
			if inpostags:
				sufixes=["_chunk(C"+str(index)+");\n"]
			else:
				sufixes=["_pos(P"+str(index)+");\n","_chunk(C"+str(index)+");\n"]
			#if index>0:
				#sufixes.append("_pos_link(POS"+str(index-1)+",POS"+str(index)+");\n")
				#sufixes.append("_chunk_link(C"+str(ind-1)+",C"+str(index)+");\n")
			for sufix in sufixes:
				if(capitalized.match(s)):
					result+="CAPITALIZED"+printind(i)+sufix
				if(caplet.match(s)):
					result+="CAPLET"+printind(i)+sufix
				if(allcaps.match(s)):
					result+="ALLCAPS"+printind(i)+sufix
				if(caps.match(s)):
					result+="CAPS"+printind(i)+sufix
				if(contains_num.search(s)):
					result+="CONTAINS_NUM"+printind(i)+sufix
				if(contains_hyp.search(s)):
					result+="CONTAINS_HYP"+printind(i)+sufix
				if(contains_appos.search(s)):
					result+="CONTAINS_APPOS"+printind(i)+sufix
				if(contains_per.search(s)):
					result+="CONTAINS_PER"+printind(i)+sufix
				if(is_num.match(s)):
					result+="IS_NUM"+printind(i)+sufix
				if(is_int.match(s)):
					result+="IS_INT"+printind(i)+sufix
				if(ends_s.search(s)):
					result+="ENDS_S"+printind(i)+sufix
				if(ends_ing.search(s)):
					result+="ENDS_ING"+printind(i)+sufix
				if(ends_ed.search(s)):
					result+="ENDS_ED"+printind(i)+sufix


	return result
		

def get_dictionary(file_name,words,postags,chunktags):
	#print docno,
	data_file = open(file_name, "r")

	rawwords={}
	
	for line in data_file:
		line = line.strip()
		if line != "":

			w = re.split("\s",line)

			w0=norm_digits(clean(w[0]).lower())
			if not w0 in rawwords:
				rawwords[w0]=1
			else:
				rawwords[w0]+=1
			postags[escape(w[1])]=1
			
			chunktags[w[2].replace("-","_")]=1
	oovcount=0
	for w,c in rawwords.iteritems():
		if c>=freq_limit:
			words[w]=c
		else:
			oovs[w]=c
			oovcount+=1
	print "Constructed dictionary:"
	#print words
	print str(len(words))+" items and "+str(oovcount)+" oovs "
	return

#Print the template data file
def print_features(feature_file_name,words,postags,chunktags):
	ff = open(feature_file_name,'w')
        #print variable types
	print >>ff, "types:"
	print >>ff, "POS := ["+",".join(postags.keys())+"]"
	print >>ff, "CHUNK := ["+",".join(chunktags.keys())+"]"
	print >>ff, ""

	print >>ff, "features:"
	#starting and ending sequence features
	print >>ff, "chunk_link(CHUNK,CHUNK):=[*,*]"
	print >>ff, "pos_chunk_link(POS,CHUNK):=[*,*]"
	print >>ff, "posm1_chunk_link(POS,CHUNK):=[*,*]"
	print >>ff, "posp1_chunk_link(POS,CHUNK):=[*,*]"
		
	print >>ff, "chunk_start(CHUNK):=[*]"
	#print >>ff, "pos_start(POS):=[*]"
	print >>ff, "chunk_end(CHUNK):=[*]"
	print >>ff, "pos_end(POS):=[*]"

	for i in wordrange:
		for suf in feat_suf:
			print >>ff, "w"+printind(i)+"_oov"+suf

	# "other" features including capitalization, etc.
	for i in featrange:
		for pre in other_features:
			for suf in feat_suf:
				print >>ff, pre+printind(i)+suf
	#lexical features
	for w, v in words.iteritems():
		for i in wordrange:
			for suf in feat_suf:
				print >>ff, "w"+printind(i)+"_"+w+suf

	print >>ff, "\nrelations:\n"#\nweights:"
#	for ct in chunktags:
#		for ct1 in chunktags:
#			if ct1[0] == 'I' and not ct[-3:]==ct1[-3:]:
#				print >>ff, "chunk_link["+ct+","+ct1+"]=-20"
def print_example(df,lines,count,features,uwords,oovcount,uoovs):
	print >>df, "//example "+str(count)
	print >>df, "example:"
	ind = 0
	maxind = len(lines)-1
	features+= "chunk_start(C0);\n"
	#features+= "pos_start(P0);\n"
	for l in lines:
		w = re.split("\s",l)

		features+= printfeats(w[0],ind,maxind)
		w0=clean(w[0])
		w0=norm_digits(w0.lower())
		if not w0 in words:
			oovcount+=1
			uoovs[w0]=1
			if w0 in oovs:
				oov_train_count+=1
			w0 = "oov"
		else:
			uwords[w0]=1
		postag = ""
		if printpostags:
			postag="="+escape(w[1])
		inp = " in" if inpostags else "";
		print >>df, "POS P"+str(ind)+postag+inp+";"
		chunktag = w[2].replace("-","_");
		if not chunktag in chunktags:
			chunktag="=O"
		else:
			chunktag="="+chunktag
		print >>df, "CHUNK C"+str(ind)+chunktag+";"

		for i in wordrange:
			index = ind+i
			if index>=0 and index<=maxind:
				if not inpostags: features+= "w"+printind(i)+"_"+w0+"_pos(P"+str(index)+");\n"
				features+= "w"+printind(i)+"_"+w0+"_chunk(C"+str(index)+");\n"
				#if index>0:
					#features+= "w"+printind(i)+"_"+w0+"_pos_link(POS"+str(index-1)+",POS"+str(index)+");\n"
					#features+= "w"+printind(i)+"_"+w0+"_chunk_link(C"+str(index-1)+",C"+str(index)+");\n"

		if ind>0:
			features+= "posp1_chunk_link(P"+str(ind)+",C"+str(ind-1)+");\n"
		features+= "pos_chunk_link(P"+str(ind)+",C"+str(ind)+");\n"
		if ind<maxind: features+= "posm1_chunk_link(P"+str(ind)+",C"+str(ind+1)+");\n"
		if ind>0:
			if not inpostags: features+="pos_link(P"+str(ind-1)+",P"+str(ind)+");\n"
			features+="chunk_link(C"+str(ind-1)+",C"+str(ind)+");\n"
		ind +=1
	features+= "chunk_end(C"+str(ind-1)+");\n"
	features+= "pos_end(P"+str(ind-1)+");\n"

	print >>df, "features:\n"+features
	features=""
	ind = 0
	return

def print_data(in_file,out_file,words,postags,chunktags):
	infile = open(in_file,'r')
	df = open(out_file,'w')

	ind = 0
	features=""
	count = 1
	lines = []
	oovcount = 0
	oov_train_count = 0
	correct = 0
	uwords = {}
	uoovs={}
	for line in infile:
		line = line.strip()
		if line == "":
			#print " ".join(lines) +"\n*********\n"
			
			print_example(df,lines,count,features,uwords,oovcount,uoovs)
			count +=1
			if count>num_examples: break
			lines = []
		else:
			lines.append(line)
	if len(lines)>1:
		print_example(df,lines,count,features,uwords,oovcount,uoovs)
	#print lines
	cont_num=0
	num_count=0
	int_count=0
#	for w in uoovs:
#		if contains_num.search(w):
#			cont_num+=1
#		if is_num.match(w):
#			num_count+=1
#		if is_int.match(w):
#			int_count+=1

		#if not w in oovs:
		#	num="true" if is_int.match(w) else "false"
		#	cnum="true" if contains_num.search(w) else "false"
		#	print w + "\tis_num="+num+" cont_num="+cnum
	print in_file+": "+str(len(uoovs))+" oovs ("+str(oovcount)+" mentions) and "+str(len(uwords))+" words; "+str(oov_train_count)+" oov mentions in train set"
	#print uwords
	#+str(cont_num)+" contians int "+str(num_count)+" are numbers "+str(int_count)+" are integers."

words = {}
postags = {}
chunktags = {}

fn = data_file.replace('train.txt','')

get_dictionary(data_file,words,postags,chunktags)

if(npsonly):
	chunktags={"B_NP":1,"I_NP":1,"O":1}


#print postags
print chunktags

designation = ".c"+str(word_context)
designation += ".npsonly" if npsonly else ".all"
if not inpostags:
	designation+=".hiddentags"


#print_features("feature.template"+designation+".ff",words,postags,chunktags)
print_features(fn+"template.ff",words,postags,chunktags)
#print_data(data_file,"train.data"+designation+".ff",words,postags,chunktags)
print_data(data_file,fn+"train.data.ff",words,postags,chunktags)
test_fn = data_file.replace('train.txt','test.txt')
#print_data(test_fn,"test.data"+designation+".ff",words,postags,chunktags)
print_data(test_fn,fn+"test.data.ff",words,postags,chunktags)
