name="hello"
server_id=10
class_id=29b46525686ccf97ef3f12bf825c56733229efd5-1648960891613
session_id=438d148bfa3cf13739c950d1a002cf83e865aa96-1648960980166
url="https://blue$server_id.aut.ac.ir/bigbluebutton/presentation/$class_id/$class_id/$session_id/svg"
tmp_dir=$(mktemp -d)
for i in {1..69}
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

rm $tmp_dir

