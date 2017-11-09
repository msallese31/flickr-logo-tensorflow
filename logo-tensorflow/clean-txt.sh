count=0
while read logo; do
  while read trainitem; do
  	trainlogo=`echo $trainitem | cut -d "," -f1`
  	# echo "$logo != $trainlogo"
  	if [[ $logo == $trainlogo ]]; then
  	  # echo "$logo == $trainlogo for item $trainitem" 
  	  replaced=${trainitem//$logo/$count} 
  	  echo $replaced >> cleaned.txt
  	fi

  done <trainset.txt
  (( count++ ))
done <logo-names.txt