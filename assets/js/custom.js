function show_password() {
    let pass_input = document.getElementById('pass-input')
    if (pass_input.type === 'password') {
        pass_input.type = 'text'
    } else {
        pass_input.type = 'password'
    }
}

function set_wall(url) {
            $(".Scriptcontent").css(`background`, `url(`+ url + `) top no-repeat`
            );
            $(".Scriptcontent").css("background-size","cover");
}
function remove_from_favorite(slug) {
    console.log(slug)
    $.ajax({
        url: '../../user/remove-favorite-movie/?slug=' + slug,
        type: 'GET',
        success: function (data) {
            if (data.success == true) {
                $("#favorite").attr('class', 'bi-bookmark-plus')
                $("#favorite").attr('onclick', 'add_to_favorite("' + slug + '")')
            }
        }
    })
}

function add_to_favorite(slug) {
    console.log(slug)
    $.ajax({
        url: '../../user/favorite-movie/?slug=' + slug,
        type: 'GET',
        success: function (data) {
            if (data.success == true) {
                $("#favorite").attr('class', 'bi-bookmark-plus-fill')
                $("#favorite").attr('onclick', 'remove_from_favorite("' + slug + '")')
            }
        }
    })
}

function hide_element(id) {
    $(`#${id}`).hide(800);
}
function show_element(id) {
    $(`#${id}`).show(800);
}