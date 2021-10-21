#DEFAULT PARAMETERS:
# locs:list=None, (default to all locations)
# tracks:list=None, (default to all tracks)
# env_class:str='NRasterizedIncrementingAgent',
# env_args:dict={width:36,height:36,m_per_px:2},
# expert_class:str='NormalizedIntersimpleExpert',  
# expert_args:dict={mu:0.001}):

cd ./src/data
python -m expert --locs='[DR_USA_Roundabout_FT]' 
cd ../