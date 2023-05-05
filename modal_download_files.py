#!/usr/bin/env python

"download files after finetuning, skip checkpoints and optimizer states"

import modal
import os

vol = modal.lookup( "test_volume" )
files = vol.listdir( '/output/**' )
print( files )

for f in files:
	if f.type != 1:
		continue
	print( f )
	
	if 'checkpoint' in f.path:
		continue
	
	if f.path.endswith( '.pt' ):
		print( "SKIPPING" )
		continue
	
	try:
		os.makedirs( os.path.dirname( f.path ))
	except FileExistsError:
		pass
	
	with open( '{}'.format( f.path ), 'wb' ) as o_f:
		for chunk in vol.read_file( f.path ):
			o_f.write( chunk )

	o_f.close()
