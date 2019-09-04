'''
Created on 7 Nov 2018

@author: thomasgumbricht
'''

__version__ = '0.3.1'
VERSION = tuple( int(x) for x in __version__.split('.') )
metadataD = { 'name':'image', 'author':'Thomas Gumbricht', 
             'author_email':'thomas.gumbricht@gmail.com',
             'title':'Image processing', 
             'label':'Specific, but generic, processes for satellite image enhancement, interpretation and classification.',
             'prerequisites':'',
             'image':'avg-trmm-3b43v7-precip_3B43_trmm_2001-2016_A'}