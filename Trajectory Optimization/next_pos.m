function P = next_pos(x,y,dir,v_user)
x_new = x;
y_new = y;
v_user_x = v_user(1);
v_user_y = v_user(2);
    switch dir
        case 0
            x_new = x_new+v_user_x;
            y_new = y_new+v_user_y;
        case 1
            x_new = x_new+v_user_x;
            y_new = y_new-v_user_y;
        case 2
            x_new = x_new-v_user_x;
            y_new = y_new-v_user_y;
        otherwise
            x_new = x_new-v_user_x;
            y_new = y_new+v_user_y;
    end
    P(1) = x_new;
    P(2) = y_new;
end