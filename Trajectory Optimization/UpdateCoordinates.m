function new_coordinates = UpdateCoordinates(coordinates, k,directions,v_user)
    new_coordinates = [];
    for i=1:k
        dir = directions(i);
        P = next_pos(coordinates(i,1),coordinates(i,2),dir,v_user(i,:));
        new_coordinates(i,1) = P(1);
        new_coordinates(i,2) = P(2); 
    end
end