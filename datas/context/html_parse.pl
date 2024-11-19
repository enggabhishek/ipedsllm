#!/usr/bin/perl

use HTML::TableExtract;
use HTML::LinkExtor;
use Data::Dumper;

## Open file 'ipeds_2022_html_fragment' which is an HTML table fragment from the IPEDS website: 
## https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx?year=2022
## This table summarizes the data files available to download for year 2022 with links to point to them
open my $fh, '<', 'ipeds_2022_html_fragment' or die "Can't open file $!";
my $file_content = do { local $/; <$fh> };

## Extract the columns from the HTML table fragment into an array
my $te = HTML::TableExtract->new( headers => ['Year','Survey','Title','Data File','Stata Data File','Programs','Dictionary'], keep_html => 1 );
$te->parse($file_content);
my $ts = $te->first_table_found();
my @columns = $ts->columns;

## Extract the link urls from the anchor tags
my @urls = ();
sub callback {
   my($tag, %attr) = @_;
   push(@urls, values %attr);
}
my $le = HTML::LinkExtor->new(\&callback);

## Column 4 has the data files in CSV format
for my $file_link (@{$columns[3]}) {
  $le->parse($file_link);
}

## Column 7 has the data dictionaries
for my $file_link (@{$columns[6]}) {
  $le->parse($file_link);
}

## Add the base URL onto the links and print output to the screen
@urls = map { 'https://nces.ed.gov/ipeds/datacenter/'.$_ } @urls;
print join("\n", @urls), "\n";
