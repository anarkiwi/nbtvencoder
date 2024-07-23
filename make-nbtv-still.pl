#!/usr/bin/perl

use strict;
use warnings;
use Image::Magick;
use Audio::Wav;

my (@infiles) = @ARGV;
my ($outfile) = "nbtv.wav";
my ($maxframes) = 100000;
my ($lines) = 32;
my ($dots) = 102;

sub pack_samples {
  my ($sample, $sample_count) = @_;
  return pack(sprintf("s%u", $sample_count), $sample);
}

sub make_still_frame_samples {
  my ($image, @pixels) = @_;
  my ($samples) = "";
  my ($i) = 0;

  for ($i = 0; $i < 16; ++$i) {
    my ($startframe_sample) = pack_samples(int(200 - rand(400)), 1);
    $samples .= $startframe_sample;
  }
  my ($x, $y);
  for ($y = 1; $y <= $lines; ++$y) {
    for ($x = 1; $x <= $dots; ++$x) {
      my ($pixel) = 1 - $image->GetPixel(x=>$x, y=>$y);
      my ($val) = int($pixel * 6000) + rand(200);
      my ($sample) = pack_samples($val, 1);
      $samples .= $sample;
    }
    for ($i = 0; $i < 6; ++$i) {
      my ($endofline_sample) = pack_samples(int(-4000 - rand(1000)), 1);
      $samples .= $endofline_sample;
    }
  }
  return $samples;
}

sub write_one_still {
  my ($infile, $wavwrite) = @_;
  my ($image) = Image::Magick->new;
  my ($err) = $image->Read($infile);
  die($err) if $err;

  $image->Quantize(colorspace=>'gray');
  $image->Flip();
  $image->Rotate(degrees=>-90);
  $image->AdaptiveResize(height=>$lines, width=>$dots);
  my (@pixels) = $image->GetPixels(map=>'I', height=>$lines, width=>$dots, normalize=>1);
  my ($samples) = make_still_frame_samples($image, @pixels);

  my ($i);
  for ($i = 0; $i < $maxframes; ++$i) {
    $wavwrite->write_raw($samples, length($samples));
  }
}

my ($wav) = Audio::Wav->new;
my ($wavwrite) = $wav->write($outfile, { bits_sample => 16,
                                         sample_rate => 44100,
                                         channels => 1 }); # must be two channels for NBSC

foreach my $infile (@infiles) {
  die("usage: $0 <image> [image...]") unless $infile && -f $infile;
  print "$infile\n";
  write_one_still($infile, $wavwrite);
}

