name="hello"
url="https://blue6.aut.ac.ir/bigbluebutton/presentation/29b46525686ccf97ef3f12bf825c56733229efd5-1646718352405/29b46525686ccf97ef3f12bf825c56733229efd5-1646718352405/0103519214f8d00b7d80329aae893be2ef3c3384-1646718390574/svg"
tmp_dir=$(mktemp -d)
for i in {1..27}
do
	if ! curl -s --head  --request GET "$url/$i" | grep "404 Not Found" > /dev/null
    then
       curl "$url/$i" > "$tmp_dir/$i.svg"
    fi
done
for file in $(ls -1v $tmp_dir/*.svg)
do
	rsvg-convert -f pdf -o "$file.pdf" $file
done

pdfunite $(ls -1v $tmp_dir/*.pdf) "$name.pdf"

# rm -r $tmp_dir
echo $tmp_dir

