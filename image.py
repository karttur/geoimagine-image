'''
Created on 7 Nov 2018

@author: thomasgumbricht
'''

import os
import numpy as np
from geoimagine.ktnumba import ImageTransform, ImageFgBg
from geoimagine.mask import MultiBandMasking
#gc = grabarge collect
import gc

class ProcessImage:
    '''class for image processing'''   
    def __init__(self, process, session, verbose):
        self.session = session
        self.verbose = verbose
        self.process = process
        if self.process.proc.processid[0:4].lower() == 'fgbg': 
            self._SetFGBGparams()
       
        for locus in self.process.srcLayerD:
            for datum in self.process.srcLayerD[locus]:
                dstLayerD = {}
                for dstcomp in self.process.dstLayerD[locus][datum]:
                    layerId = self.process.dstLayerD[locus][datum][dstcomp].comp.id                    
                    if not self.process.dstLayerD[locus][datum][dstcomp]._Exists() or self.process.overwrite:
                        dstLayerD[layerId] = self.process.dstLayerD[locus][datum][dstcomp]
                    else:
                        self.session._InsertLayer(self.process.dstLayerD[locus][datum][dstcomp], self.process.overwrite, self.process.delete)

                if len(dstLayerD) == 0:
                    continue
                srcLayerD = {}
                runFlag = True
                for srccomp in self.process.srcLayerD[locus][datum]:
                    if not (self.process.srcLayerD[locus][datum][srccomp]):
                        print ('composition missing',datum)
                        runFlag = False
                        continue
                    if not os.path.exists(self.process.srcLayerD[locus][datum][srccomp].FPN):
                        print ('composition file missing',datum)
                        runFlag = False
                        continue
                    layerId = self.process.srcLayerD[locus][datum][srccomp].comp.id
                    self.process.srcLayerD[locus][datum][srccomp].ReadRasterLayer()
                    #transfer the layer to a local id based variable
                    srcLayerD[layerId] = self.process.srcLayerD[locus][datum][srccomp]
                    print (srccomp,layerId,self.process.srcLayerD[locus][datum][srccomp].FPN)
                if runFlag:                         
                    if self.process.proc.processid[0:15].lower() == 'lineartransform':  
                        self._LinearTransformMODISSingleTile(srcLayerD, dstLayerD)
    
                    elif self.process.proc.processid[0:4].lower() == 'fgbg':  
    
                        self._fgbgmodisSingleTile(srcLayerD, dstLayerD)
    
                    else:
                        exitStr = 'EXITING, unknown process in ProcessImage (image.py: %(s)s)' %{'s':self.process.proc.processid}
                        exit(exitStr)
                    #Write the results
      
    def _SetFGBGparams(self): 
        #TGTODO ASSEMBEL ALL PARAMETERS IN A SEPARATE CLASS
        from math import atan, sin, cos
        breakFlag = False
        srcKeys = False
        for locus in self.process.srcLayerD:
            if breakFlag:
                break
            for datum in self.process.srcLayerD[locus]:
                dstLayerD = {}
                for dstcomp in self.process.dstLayerD[locus][datum]:
                    layerId = self.process.dstLayerD[locus][datum][dstcomp].comp.id
                    dstLayerD[layerId] = []
                srcLayerD = {}
                for srccomp in self.process.srcLayerD[locus][datum]:
                    if not (self.process.srcLayerD[locus][datum][srccomp]):
                        continue
                    layerId = self.process.srcLayerD[locus][datum][srccomp].comp.id
                    srcLayerD[layerId] = []
                print (locus,datum,'keys',list(srcLayerD.keys()) )
                if len(list(srcLayerD.keys())) == 2:
                    srcKeys = list(srcLayerD.keys())
                    breakFlag = True
                '''
                if (self.process.srcLayerD[locus][datum][srccomp]):
                    break
                '''
        '''        
        srcKeys = list(srcLayerD.keys())
        
        if len(srcKeys) == 0 and self.process.params.acceptmissing:
            return False
        '''
        if not srcKeys:
            exit('No input bands found for ImageProcess: FGBG ')       
        if 'xband' in srcKeys:
            self.xb = 'xband'
        elif 'bandx' in srcKeys:
            self.xb = 'bandx'
        else:
            print (srcKeys)
            exitstr= 'the fgbg process must have either xband or bandx as srcband id %s' %(srcKeys)
            exit(exitstr)
        if 'yband' in srcKeys:
            self.yb = 'yband'
        elif 'bandy' in srcKeys:
            self.yb = 'bandy'
        else:
            exitstr= 'the fgbg process must have either yband or bandy as srcband id'
            exit(exitstr)
        dstKeys = list(dstLayerD.keys())
        if 'fg' or 'bg' in dstKeys:
            if 'fg' in dstKeys:
                self.fg = 'fg'
            else:
                self.fg = False
            if 'bg' in dstKeys:
                self.bg = 'bg'
            else:
                self.bg = False
        else:
            exitstr= 'the fgbg process must have either fg or bg (or both) as dstband id'
            exit(exitstr)
        
        #Do the rotation
        angrad = -atan(self.process.params.slope)
        rangdeg = 180 * angrad / 3.1415
        rangdeg += 45
        #Convert degrees to radians
        rangrad = 3.1415 * rangdeg / 180
        #Get the sin and cos angles
        self.sinrang = sin(rangrad) 
        self.cosrang = cos(rangrad)
                              
    def _LinearTransformMODISSingleTile(self, srcLayerD, dstLayerD):
        '''
        '''
        srcKeys = list(srcLayerD.keys())
        for dst in dstLayerD:
            #Create the dst layer
            dstLayerD[dst].layer = lambda:None
            #Set the np array as the band
            dstLayerD[dst].layer.NPBAND = np.zeros(srcLayerD[srcKeys[0]].layer.NPBAND.shape) 
            print ('lineartransfrom to %s' %(dstLayerD[dst].FPN))

        #then run
        for dst in dstLayerD:
            for src in srcLayerD:
                xid = '%s%s' %(srcLayerD[src].comp.id, 
                               dstLayerD[dst].comp.id) 
                scalefacD = getattr(self.process.proc.transformscale, xid)
                scalefac = scalefacD['scalefac']
                offsetD = getattr(self.process.proc.transformoffset, srcLayerD[src].comp.id)
                offset = offsetD['offset']
                #ImageTransform is a numba JIT function
                dstLayerD[dst].layer.NPBAND = ImageTransform(dstLayerD[dst].layer.NPBAND,srcLayerD[src].layer.NPBAND,offset,scalefac)

        MultiBandMasking(srcLayerD, dstLayerD)
        
        for dst in dstLayerD:
            #copy the geoformat from the src layer
            dstLayerD[dst].CopyGeoformatFromSrcLayer(srcLayerD[srcKeys[0]].layer)
            #write the results
            dstLayerD[dst].CreateDSWriteRasterArray()
            #Register the layer
            self.session._InsertLayer(dstLayerD[dst], self.process.overwrite, self.process.delete)
            dstLayerD[dst].layer.NPBAND = None
            dstLayerD[dst] = None
        for src in srcLayerD:
            srcLayerD[src].layer.NPBAND = None
            srcLayerD[src] = None  
        #grabage collect
        gc.collect()
                       
    def _fgbgmodisSingleTile(self, srcLayerD, dstLayerD):
        '''
        '''

        X = srcLayerD[self.xb].layer.NPBAND
        Y = srcLayerD[self.yb].layer.NPBAND
        print (self.xb,'self.xb')
        print (self.yb,'self.yb')
        
        if self.fg:
            dstLayerD['fg'].layer = lambda:None
            print ('foregrounding to %s' %(dstLayerD['fg'].FPN))
            #Set the np array as the band
            #self.process.dstLayerD[locus][datum][dstComp].layer.NPBAND = dstBAND
            #dstLayerD[dst].layer.NPBAND = np.zeros(srcLayerD[srcKeys[0]].layer.NPBAND.shape) 

            dstLayerD['fg'].layer.NPBAND = ImageFgBg(self.process.params.rescalefac, 
                    self.sinrang, self.cosrang, X, Y, self.process.params.intercept, self.process.params.calibfac)

            #dstLayerD['fg'].layer.NPBAND = self.process.params.rescalefac * ((self.sinrang*(x+y-self.process.params.intercept) + self.cosrang*(-x+y-self.process.params.intercept)) / 
            #                 (self.sinrang*(x-y+self.process.params.intercept) + self.cosrang*( x+y-self.process.params.intercept) + self.process.params.calibfac ))
        #FG =  5942*( ( self.sinrang*(x+y+2080) + self.cosrang*(-x+y+2080) ) / ( self.sinrang*(x - y - 2080)+self.cosrang*(x+y+2080) + 7000 ) )
        #twi = 5942*( ( _sinrang*(x+y+2080) + _cosrang*(-x+y+2080) ) / ( _sinrang*(x - y - 2080)+_cosrang*(x+y+2080) + 7000 ) )
        if self.bg:
            dstLayerD['bg'].layer = lambda:None
            print ('backgrounding to %s' %(dstLayerD['bg'].FPN))
            
            #BG = self.process.rescalefac * ((self.sinrang*(x+y+self.process.intercept) + self.cosrang*(-x+y+self.process.intercept)) / 
            #                     (self.sinrang*(x-y-self.process.intercept) + self.cosrang*( x+y+self.process.intercept) + self.process.calibfac ))
        MultiBandMasking(srcLayerD, dstLayerD)

        for dst in dstLayerD:
            #copy the geoformat from the src layer
            dstLayerD[dst].CopyGeoformatFromSrcLayer(srcLayerD[self.xb].layer)
            #write the results
            dstLayerD[dst].CreateDSWriteRasterArray()
            #Register the layer
            self.session._InsertLayer(dstLayerD[dst], self.process.overwrite, self.process.delete)
            dstLayerD[dst].layer.NPBAND = None
            dstLayerD[dst] = None
 
        for src in srcLayerD:
            srcLayerD[src].layer.NPBAND = None
            srcLayerD[src] = None  
        #grabage collect
        gc.collect()  


def SetFGBGparams(slope): 
        #TGTODO ASSEMBEL ALL PARAMETERS IN A SEPARATE CLASS
        from math import atan, sin, cos
        
        angrad = -atan(slope)
        rangdeg = 180 * angrad / 3.1415
        rangdeg += 45
        #Convert degrees to radians
        rangrad = 3.1415 * rangdeg / 180
        #Get the sin and cos angles
        sinrang = sin(rangrad) 
        cosrang = cos(rangrad)
        
        return (sinrang, cosrang)
     
def TWItest(rescalefac, sinrang, cosrang, X, Y, intercept, calibfac):
    A = rescalefac * ( (sinrang*(X+Y-intercept) + cosrang*(-X+Y-intercept)) / 
                             (sinrang*(X-Y+intercept) + cosrang*( X+Y-intercept) + calibfac ) )
    return A
        
           
           
if __name__ == "__main__":
    slope = 1.6
    intercept = -2080
    rescalefac = 5942
    calibfac = 7000
    #X = soil
    #Y = wet
    X = 2036
    Y = -606
    X = 5000
    Y = 5000
    
    sinrang, cosrang = SetFGBGparams(slope)
    
    print (TWItest(rescalefac, sinrang, cosrang, X, Y, intercept, calibfac))