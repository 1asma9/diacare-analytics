// live values next to the sliders + example buttons
document.querySelectorAll('input[type=range]').forEach(function(r){
  var out=document.getElementById(r.id+'_v'); var sync=function(){out.textContent=r.value}; r.addEventListener('input',sync); sync();
});
document.querySelectorAll('[data-example]').forEach(function(b){
  b.addEventListener('click',function(){
    var d=JSON.parse(b.getAttribute('data-example'));
    Object.keys(d).forEach(function(k){var el=document.getElementById(k); if(el){el.value=d[k]; el.dispatchEvent(new Event('input'));}});
  });
});
var f=document.getElementById('form'); if(f){f.addEventListener('submit',function(){var s=document.getElementById('go'); s.disabled=true; s.textContent='Calculating...';});}
