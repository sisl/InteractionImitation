# cp-videos videos/ videos/icra23/

agents=( 5 27 39 43 47 53 63 81 83 87 93 96 105 113 124 127 130 134 )

for a in "${agents[@]}"
do
    cp "$1/expert_agent/loc0/track0/agent${a}_ani.mp4" "$2/t${a}expert.mp4"
    cp "$1/idm/loc0/track0/agent${a}_ani.mp4" "$2/t${a}idm.mp4"
    cp "$1/shail/loc0/track0/agent${a}_ani.mp4" "$2/t${a}shail.mp4"
done
